import sys
import os
from joblib import Parallel, delayed
sys.path.append('../H2_SurfaceMatch')
import utils.input_output as input_output
import utils.utils as ut
import SRNF_match as matching
from scipy.spatial import cKDTree
import open3d as o3d
import numpy as np
import gc
import torch
import pymeshlab
import pyvista as pv
import pyacvd
import time
import trimesh

down_vert = int(1000)

mesh_folder = "reg_pcs_spider"
pca_folder = "pca_srnf_spider"
final_folder = "mesh_final_spider"

def print_elapsed(start, end):
	elapsed = end - start
	
	hours = int(elapsed // 3600)
	
	minutes = int((elapsed % 3600) // 60)
	
	seconds = int(elapsed % 60)
	
	print(f"[Progress] Elapsed time: {hours}h {minutes}m {seconds}s", flush = True)

import numpy as np

def procrustes_align(X, mean, use_scale=True):
    """
    X:    (N,3) point set
    mean: (N,3) reference mean shape
    use_scale: if True, perform similarity Procrustes;
               if False, perform rigid Procrustes.

    Returns:
        X_aligned: aligned points
        R: rotation matrix
        t: translation vector
        s: scale factor (1.0 if use_scale=False)
    """

    # centroids
    X_centroid = X.mean(axis=0)
    Y_centroid = mean.mean(axis=0)

    # center
    Xc = X - X_centroid
    Yc = mean - Y_centroid

    # covariance
    H = Xc.T @ Yc

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # rotation
    R = Vt.T @ U.T

    # prevent reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # optional scaling
    if use_scale:
        s = np.sum(S) / np.sum(Xc**2)
    else:
        s = 1.0

    # translation
    t = Y_centroid - s * (X_centroid @ R)

    # aligned points
    X_aligned = s * (X @ R) + t

    return X_aligned, R, t, s

def pca_icp(target, mean, mean_tris, scale=True):
    # Source mesh
    mean = trimesh.Trimesh(vertices=mean, faces=mean_tris)
    target = trimesh.Trimesh(vertices=target, faces=mean_tris)

    # ICP with similarity transform
    T, _ = trimesh.registration.mesh_other(
        mean,
        target,
        scale=scale,
        reflection=False
    )

    target.apply_transform(T)

    aligned_points = target.vertices

    A = T[:3, :3]
    t = T[:3, 3]

    # Uniform scale
    s = np.cbrt(np.linalg.det(A))

    # Rotation
    R = A / s

    return aligned_points, R, t, s

def icp(target, mean):
    mean_pc = o3d.geometry.PointCloud()
    mean_pc.points = o3d.utility.Vector3dVector(np.asarray(mean))

    target_pc = o3d.geometry.PointCloud()
    target_pc.points = o3d.utility.Vector3dVector(np.asarray(target))

    # Source mesh
    distance_threshold = 30
    
    icp_result = o3d.pipelines.registration.registration_icp(
    target_pc, mean_pc, distance_threshold, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    T = icp_result.transformation
    
    target_pc = target_pc.transform(T)
    aligned_points = np.asarray(target_pc.points)

    A = T[:3, :3]
    t = T[:3, 3]

    # Uniform scale
    s = np.cbrt(np.linalg.det(A))

    # Rotation
    R = A / s

    return aligned_points, R, t, s


def get_normals(vertices, triangles):
    # print("Obtaining Normals", flush=True)
    vertices = np.asarray(vertices)
    faces = np.asarray(triangles)
    
    n_vertices = vertices.shape[0]
    mask = np.all((faces >= 0) & (faces < n_vertices), axis=1)
    faces = faces[mask]
    
    pymesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymesh)
    # ms.meshing_re_orient_faces_by_geometry(rays = 100)
    ms.compute_normal_per_vertex(weightmode=0)
    ml_mesh = ms.current_mesh()
    normals = np.array(ml_mesh.vertex_normal_matrix(), dtype=np.float64)
    
    return normals
    
def acvd_resample(mesh, clus_num):
    # print("Resampling Mesh", flush=True)
    V, F = mesh
    faces_pv = np.hstack([np.full((F.shape[0], 1), 3), F]).flatten()
    pv_mesh = pv.PolyData(V, faces_pv)
    # pv_mesh.plot(color='w', show_edges=True)
    clus = pyacvd.Clustering(pv_mesh)
    # mesh is not dense enough for uniform remeshing
    # clus.subdivide(3)
    clus.cluster(clus_num)
    # plot clustered
    # clus.plot()

    remesh = clus.create_mesh()
    remesh.clean(inplace=True)

    # plot
    # remesh.plot(color='w', show_edges=True)

    new_V = remesh.points
    new_F = remesh.faces.reshape((-1, 4))[:, 1:]

    return new_V, new_F
    

def parallel_matching(i, j, mean, target_folder, target_path, label, intermediate_folder, R, t, s, start):
    target_name = os.path.splitext(os.path.basename(target_path))[0].split(".")[0]
    # print(f"Started Registration of {target_name} on iteration {j}", flush = True)

    [VT,FT,FunT] = input_output.loadData(os.path.join(intermediate_folder,target_path))
    

    target = [VT,FT]
    
    if label in [41,42,43,44]:
        sig_geom_val1 = .2
        sig_geom_val2 = .1
        tri_num = int(FT.shape[0]*15/16)
    elif label in [45]:
        sig_geom_val1 = .3
        sig_geom_val2 = .1
        tri_num = int(FT.shape[0]*15/16)
    elif label in [91,92,93,94,95]:
        sig_geom_val1 = .1
        sig_geom_val2 = .1
        tri_num = int(FT.shape[0]*15/16)
    elif label in [100]:
        sig_geom_val1 = .2
        sig_geom_val2 = .2
        tri_num = int(FT.shape[0]*15/16)
    
    target_init = [mean[0],mean[1]]
    
    # Set parameters for the two successive runs
    weight_MCV=0 #no SRCF term = 0
    
    parameters1 = {'weight_MCV':weight_MCV,'weight_coef_dist_S': 10**3, 'weight_coef_dist_T': 10**3,\
              'kernel_geom':'gaussian','sig_geom': sig_geom_val1,'kernel_grass':'binet',\
              'sig_grass':1 ,'max_iter': 100,'use_fundata':0}
    # parameters2 = {'weight_MCV':weight_MCV,'weight_coef_dist_S': 10**4, 'weight_coef_dist_T': 10**4,\
    #           'kernel_geom':'gaussian','sig_geom': sig_geom_val2,'kernel_grass':'binet',\
    #           'sig_grass': 1,'max_iter': 10,'use_fundata':0}

    paramlist=[parameters1
            #    ,parameters2
              ]
    
    num_params = len(paramlist)
    
    # Run standard matching
    # print("Performing Registration", flush=True)
    f1,fopt,Dic = matching.StandardMatching(mean,target,target_init,parameters1, type="SRNF")
    
    match_target=[f1,mean[1]]
    
    if i % 100 == 0:
        # Plot matching result 
        save_folder = target_folder + "_sourcematch"
        os.makedirs(save_folder, exist_ok=True)
        input_output.saveData(os.path.join(save_folder, target_name),"ply",
                              mean[0],mean[1],Rho=None,color=None)
        save_folder = target_folder + "_targetmatch"
        os.makedirs(save_folder, exist_ok=True)
        input_output.saveData(os.path.join(save_folder, target_name),"ply",
                              match_target[0],match_target[1],Rho=None,color=None)
        save_folder = target_folder + "_matchingplot"
        os.makedirs(save_folder, exist_ok=True)
        save_file = os.path.join(save_folder,target_name)
        input_output.plotMatchingResult(target,match_target,mean,'Standard',matched_source=None,file_name=save_file)

        # Compute geodesic 
        Geod = ut.LinearInterpolation(mean[0],match_target[0],4)
        
        # Plot geodesic
        save_folder = target_folder + "_geodesic"
        os.makedirs(save_folder, exist_ok=True)
        save_file = os.path.join(save_folder,target_name)
        input_output.plotGeodesic(Geod,mean[1],source=None,target=target,file_name=save_file,stepsize=3)
    
    # match_target_aligned, R, t, s = procrustes_align(match_target[0], mean[0], use_scale=False)
    # match_target_aligned, R, t, s = pca_icp(match_target[0], mean[0], mean[1], scale=True)
    match_target_aligned, R, t, s = icp(match_target[0], mean[0])
    # match_target_aligned = match_target[0]

    input_output.saveData(os.path.join(intermediate_folder,target_name),'ply',match_target_aligned,mean[1])

    norms = get_normals(match_target_aligned, mean[1])

    frob = np.linalg.norm(mean[0] - match_target_aligned, ord="fro")

    if i % 100 == 0 and i>0:
        print(f"Completed Registration of {i} targets on iteration {j}", flush = True)
        end = time.time()
        print_elapsed(start, end)
    # print(f"Completed Registration of {target_name} on iteration {j}", flush = True)

    return match_target_aligned, target_path, frob, R, t, s, norms

labels = [
		41, 
        42, 43, 44, 45
        ,
        91, 92, 93, 94, 95, 100
		]



if 45 in labels:
    component = "vertebra"
elif 91 in labels:
    component = "disc"
elif 291 in labels:
    component = "spinal canal"



for label in labels:
    j = 0
    max_iter = 100
    out_folder = os.path.join(pca_folder, f"vertebra_{label}")
    # out_folder = os.path.join(pca_folder, f"test")
    intermediate_folder = os.path.join(final_folder, f"vertebra_{label}")
    mean_folder = os.path.join(out_folder, "means")
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(intermediate_folder, exist_ok=True)
    os.makedirs(mean_folder, exist_ok=True)

    target_folder = f"{mesh_folder}/vertebra{label}"
    # target_folder = f"{mesh_folder}/test"
    os.makedirs(out_folder, exist_ok=True)
    
    target_paths= sorted(os.listdir(target_folder))

    sample_size = len(target_paths)
    print(sample_size)

    print("Processing meshes", flush=True)
    for target_path in target_paths:
        [VT,FT,FunT] = input_output.loadData(os.path.join(target_folder,target_path))
        target = [VT, FT]
        VT, FT = acvd_resample(target, down_vert)
        target = [VT, FT]
        input_output.saveData(os.path.join(intermediate_folder,target_path.replace('.ply','')),'ply',VT,FT)
    print("Finished processing meshes", flush=True)

    mean_path = target_paths[-1]

    [VT,FT,FunT] = input_output.loadData(os.path.join(intermediate_folder,mean_path))

    mean = [VT, FT]

    print(mean_path)
    print(VT.shape)
    print(FT.shape)
    
    tol = .1
    mean_frobs = [1e10, 1e9]
    Rs_total = [np.eye(3) for _ in range(len(target_paths))]
    ts_total = [np.zeros(3) for _ in range(len(target_paths))]
    ss_total = [1.0 for _ in range(len(target_paths))]

    while np.abs(mean_frobs[-1]-mean_frobs[-2]) > tol and j < max_iter:
        input_output.saveData(os.path.join(mean_folder, f"mean_{j}"),'ply',mean[0],mean[1])

        ps = np.empty((sample_size, down_vert*3), dtype=np.float64)
        norms = np.empty((sample_size, down_vert*3), dtype=np.float64)
        paths = []
        frob = np.empty(sample_size)
        Rs = np.empty((sample_size, 3, 3))
        ts = np.empty((sample_size, 3))
        ss = np.empty(sample_size)

        start = time.time()
        results = Parallel(n_jobs=4)(delayed(parallel_matching)(i, j, mean, target_folder, target_path, label, intermediate_folder, R, t, s, start) for i, (target_path, R, t, s) in enumerate(zip(target_paths, Rs_total, ts_total, ss_total)))

        for iter, r in enumerate(results):
            ps[iter] = r[0].reshape(-1)
            paths.append(r[1])
            frob[iter] = r[2]
            Rs[iter] = r[3]
            ts[iter] = r[4]
            ss[iter] = r[5]
            norms[iter] = r[6].reshape(-1)
        
        for iter, (r,t, s) in enumerate(zip(Rs, ts, ss)):
            Rs_total[iter] = Rs_total[iter] @ r
            ss_total[iter] *= s
            ts_total[iter] = s * (ts_total[iter] @ r) + t

        # ps = np.stack(ps, axis=0)
        np.save(os.path.join(out_folder, "ps.npy"), ps)

        mean_V = np.mean(ps, axis=0).reshape(-1,3)

        mean_frobs.append(np.linalg.norm(mean[0] - mean_V, ord="fro"))
        print(f"Difference in Means at iteration {j} is {mean_frobs[-1]}", flush=True)

        np.save(os.path.join(out_folder,"mean.npy"), mean_V)

        mean = [mean_V, FT]

        j = j + 1

        # norms = np.stack(norms, axis=0)
        np.save(os.path.join(out_folder, "norms.npy"), norms)        
        np.save(os.path.join(out_folder,"paths.npy"), paths)

        torch.cuda.empty_cache()
        
    input_output.saveData(os.path.join(mean_folder, f"mean_{j}"),'ply',mean[0],mean[1])
    input_output.saveData(os.path.join(out_folder, f"mean"),'ply',mean[0],mean[1])
    np.save(os.path.join(out_folder, "Rs_procrustes.npy"), Rs_total)
    np.save(os.path.join(out_folder, "ss_procrustes.npy"), ss_total)
    np.save(os.path.join(out_folder, "ts_procrustes.npy"), ts_total)

