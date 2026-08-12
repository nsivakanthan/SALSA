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

down_vert = int(1000)


import numpy as np

def procrustes_align(X, mean):
    """
    X:    (N,3) point set
    mean: (N,3) reference mean shape

    Returns:
        X_aligned: aligned points
        R: rotation matrix
        t: translation vector
        s: scale factor
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

    # optimal uniform scale
    s = np.sum(S) / np.sum(Xc**2)

    # translation
    t = Y_centroid - s * (X_centroid @ R)

    # aligned
    X_aligned = s * (X @ R) + t

    return X_aligned, R, t, s

def get_normals(vertices, triangles):
    vertices = np.asarray(vertices)
    faces = np.asarray(triangles).astype(np.int32, copy=False)
    
    n_vertices = vertices.shape[0]
    mask = np.all((faces >= 0) & (faces < n_vertices), axis=1)
    faces = faces[mask]
    
    pymesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymesh)
    ms.meshing_re_orient_faces_by_geometry(rays = 2000)
    ms.compute_normal_per_vertex(weightmode=0)
    ml_mesh = ms.current_mesh()
    normals = np.array(ml_mesh.vertex_normal_matrix(), dtype=np.float64)
    
    return normals
    
def acvd_resample(mesh, clus_num):
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
    

def parallel_matching(i, j, mean, target_folder, target_path, label, intermediate_folder, R, t):
    target_name = os.path.splitext(os.path.basename(target_path))[0].split(".")[0]
    
    [VT,FT,FunT] = input_output.loadData(os.path.join(target_folder,target_path))
    
    target = [VT, FT]
    
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
              'sig_grass':1 ,'max_iter': 300,'use_fundata':0}


    paramlist=[parameters1
            #    ,parameters2
              ]
    
    num_params = len(paramlist)
    
    # Run standard matching 
    f1,fopt,Dic = matching.StandardMatching(mean,target,target_init,parameters1, type="SRNF")
    
    match_target=[f1,mean[1]]
    
    if i % 100 == 0:
        # Plot matching result (press q to exit window)
        save_folder = target_folder + "_sourcematch_all"
        os.makedirs(save_folder, exist_ok=True)
        input_output.saveData(os.path.join(save_folder, target_name),"ply",
                              mean[0],mean[1],Rho=None,color=None)
        save_folder = target_folder + "_targetmatch_all"
        os.makedirs(save_folder, exist_ok=True)
        input_output.saveData(os.path.join(save_folder, target_name),"ply",
                              match_target[0],match_target[1],Rho=None,color=None)
        save_folder = target_folder + "_matchingplot_all"
        os.makedirs(save_folder, exist_ok=True)
        save_file = os.path.join(save_folder,target_name)
        input_output.plotMatchingResult(target,match_target,mean,'Standard',matched_source=None,file_name=save_file)

        # Compute geodesic 
        Geod = ut.LinearInterpolation(mean[0],match_target[0],4)
        
        # Plot geodesic
        save_folder = target_folder + "_geodesic_all"
        os.makedirs(save_folder, exist_ok=True)
        save_file = os.path.join(save_folder,target_name)
        input_output.plotGeodesic(Geod,mean[1],source=None,target=target,file_name=save_file,stepsize=3)
    
    # match_target_aligned, R, t, s = procrustes_align(match_target[0], mean[0])
    match_target_aligned, R, t, s = icp(match_target[0], mean[0])

    input_output.saveData(os.path.join(intermediate_folder,target_name),'ply',match_target_aligned,mean[1])

    norms = get_normals(match_target_aligned, mean[1])

    frob = np.linalg.norm(mean[0] - match_target_aligned, ord="fro")

    for i % 100 == 0 and i>0:
        print(f"Completed Registration of 100 targets on iteration {j}", flush = True)

    return match_target_aligned, target_path, frob, R, t, s, norms

# must do vertebra and discs separately in two runs
labels = [
		41, 42, 43, 44, 45
        # 91, 92, 93, 94, 95, 100
		]

means = []
means_norms = []
sample_sizes = []
all_ps = []
all_norms = []
all_paths = []

if 41 in labels:
    component = "vertebra"
elif 91 in labels:
    component = "disc"
elif 291 in labels:
    component = "spinal canal"

target_folder = f"reg_pcs/vertebra{labels[0]}"
target_paths= sorted(os.listdir(target_folder))
final_out_folder = f"pca_rsna/all_{component}"
intermediate_folder = f"mesh_final/vertebra_{component}"
mean_folder = os.path.join(final_out_folder, "means")
os.makedirs(final_out_folder, exist_ok=True)
os.makedirs(intermediate_folder, exist_ok=True)
os.makedirs(mean_folder, exist_ok=True)


print("Processing meshes", flush=True)
for target_path in target_paths:
    [VT,FT,FunT] = input_output.loadData(os.path.join(target_folder,target_path))
    target = [VT, FT]
    VT, FT = acvd_resample(target, down_vert)
    target = [VT, FT]
    input_output.saveData(os.path.join(intermediate_folder,target_path.replace('.ply','')),'ply',VT,FT)
print("Finished processing meshes", flush=True)

mean_path = target_paths[-1]

[VT,FT,FunT] = input_output.loadData(os.path.join(target_folder,mean_path))

mean = [VT, FT]

print(mean_path)
print(VT.shape)
print(FT.shape)

for label in labels:
    j = 0
    max_iter = 100

    target_folder = f"reg_pcs/vertebra{label}"
    out_folder = os.path.join(final_out_folder, f"vertebra_{label}")
    os.makedirs(out_folder, exist_ok=True)
    
    target_paths= sorted(os.listdir(target_folder))
    
    tol = .01
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

        results = Parallel(n_jobs=4)(delayed(parallel_matching)(i, j, mean, target_folder, target_path, label, intermediate_folder, R, t) for i, (target_path, R, t) in enumerate(zip(target_paths, Rs_total, ts_total)))

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

        ps = np.stack(ps, axis=0)
        np.save(os.path.join(out_folder, "ps.npy"), ps)
        all_ps.append(os.path.join(out_folder, "ps.npy"))

        
        mean_V = np.mean(ps, axis=0).reshape(-1,3)

        mean_frobs.append(np.linalg.norm(mean[0] - mean_V, ord="fro"))
        print(f"Difference in Means at iteration {j} is {mean_frobs[-1]}", flush = True)

        means.append(mean_V)
        np.save(os.path.join(out_folder,"mean.npy"), mean)

        mean = [mean_V, FT]

        j = j + 1

        del ps

        norms = np.stack(norms, axis=0)
        np.save(os.path.join(out_folder, "norms.npy"), norms)
        all_norms.append(os.path.join(out_folder, "norms.npy"))
        
        mean_norms = np.mean(norms, axis=0)
        means_norms.append(mean_norms)
        
        del mean, norms, mean_norms
        
        
        np.save(os.path.join(out_folder,"paths.npy"), paths)
        all_paths.append(os.path.join(out_folder,"paths.npy"))
        sample_sizes.append(len(paths))
        
        
        del paths
        torch.cuda.empty_cache()

    np.save(os.path.join(out_folder, "Rs_procrustes.npy"), Rs_total)
    np.save(os.path.join(out_folder, "ss_procrustes.npy"), ss_total)
    np.save(os.path.join(out_folder, "ts_procrustes.npy"), ts_total)


means = [np.array(m) for m in means]
means_norms = [np.array(m) for m in means_norms]
sample_sizes = np.array(sample_sizes)
total_n = sample_sizes.sum()
combined_mean = sum(n * m for n, m in zip(sample_sizes, means)) / total_n
combined_norms = sum(n * m for n, m in zip(sample_sizes, means_norms)) / total_n

del means, means_norms

np.save(os.path.join(final_out_folder,"mean.npy"), combined_mean)
np.save(os.path.join(final_out_folder,"mean_norms.npy"), combined_norms)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(combined_mean.reshape(-1,3))
pcd.normals = o3d.utility.Vector3dVector(combined_norms.reshape(-1,3))

o3d.io.write_point_cloud(os.path.join(final_out_folder,"mean.ply"), pcd)

del combined_mean, combined_norms
    
all_ps = np.concatenate([np.load(f) for f in all_ps], axis=0)
np.save(os.path.join(final_out_folder, "ps.npy"), all_ps)
del all_ps

all_norms = np.concatenate([np.load(f) for f in all_norms], axis=0)
np.save(os.path.join(final_out_folder, "norms.npy"), all_norms)
del all_norms

all_paths = np.concatenate([np.load(f) for f in all_paths], axis=0)
np.save(os.path.join(final_out_folder, "paths.npy"), all_paths)
del all_paths