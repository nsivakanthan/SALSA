import os
import argparse
import glob
import shutil
import time
import sys
# sys.path.append('./H2_SurfaceMatch')
# import utils.input_output as input_output
# import SRNF_match as matching

from joblib import Parallel, delayed
import numpy as np
import open3d as o3d
from scipy.optimize import linear_sum_assignment
import nibabel as nib
import pymeshlab
from scipy.spatial import cKDTree

def print_elapsed(start, end):
	elapsed = end - start
	
	hours = int(elapsed // 3600)
	
	minutes = int((elapsed % 3600) // 60)
	
	seconds = int(elapsed % 60)
	
	print(f"[Progress] Elapsed time: {hours}h {minutes}m {seconds}s")

def nn_spacing(points):
	tree = cKDTree(points)
	dists, _ = tree.query(points, k=2)  # nearest neighbor (excludes self)
	return np.mean(dists[:, 1])

def clean_inputs(xyz, label):
	# remove meshes with low number of points
	if remove_data(xyz) < 5:
		return None

	base_name = os.path.splitext(os.path.basename(xyz))[0].replace(f"_vertebra{label}","").replace(".nii","")
	
	file_name = "*" + base_name + ".nii.gz"
	
	file_path = glob.glob(os.path.join(os.getcwd(),'**',file_name), recursive=True)
		
	mesh = o3d.io.read_triangle_mesh(xyz)

	length = len(mesh.vertices)
	
	# set vertex threshold
	if label in np.arange(1.0, 45.0, 1.0):
		min_len = 700
	elif label == 45.0:
		min_len = 700
	else:
		min_len = 700
	
	if length < min_len:
		return None

	return xyz
	
def remove_data(xyz):
	folder_path = os.path.dirname(xyz)
	all_files = os.listdir(folder_path)
	
	return len(all_files)
	
	

def normalize_points(pc):
	p = np.asarray(pc.points)
	centroid = p.mean(axis = 0)
	centered = p - centroid
	
	rms = np.sqrt((centered ** 2).sum() /len(p))
	
	if rms > 0:
		p_norm = centered/rms
	else:
		p_norm = centered
	
	pc = o3d.geometry.PointCloud()
	pc.points = o3d.utility.Vector3dVector(p_norm)
		
	return pc, rms


def upscale(pc, rms):
	p = np.asarray(pc.points)
	norms = np.asarray(pc.normals)

	if rms > 0:
		p_scale = p * rms
	else:
		p_scale = p

	pc = o3d.geometry.PointCloud()
	pc.points = o3d.utility.Vector3dVector(p_scale)
	pc.normals = o3d.utility.Vector3dVector(norms)

	return pc


def hungarian(ply, target_p):
    # perform hungarian/jv algorithm to obtain new orderings of points 
	print("[Progress] Point Matching")
	start_hungarian = time.time()
	source_pc = o3d.io.read_point_cloud(ply)
	source_p = np.asarray(source_pc.points)
	source_norms = np.asarray(source_pc.normals)
	
	cost_matrix = np.linalg.norm(source_p[:,None,:] - target_p[None,:,:], axis = 2)
	
	row_ind, col_ind = linear_sum_assignment(cost_matrix)
	
	inverse_map = np.zeros_like(col_ind)
	inverse_map[col_ind] = row_ind
	
	new_source_p = source_p[inverse_map]
	new_source_norms = source_norms[inverse_map]
	
	new_source_pc = o3d.geometry.PointCloud()
	new_source_pc.points = o3d.utility.Vector3dVector(new_source_p)
	new_source_pc.normals = o3d.utility.Vector3dVector(new_source_norms)
	
	o3d.io.write_point_cloud(ply, new_source_pc)

	end_hungarian = time.time()
	print_elapsed(start_hungarian, end_hungarian)
	
	#p = np.array(new_source_pc.points)
	
	return new_source_p, new_source_norms



def get_normals(mesh):
    # get accurate normals and clean shapes after segmentation
	vertices = np.asarray(mesh.vertices)
	faces = np.asarray(mesh.triangles).astype(np.int32, copy=False)

	n_vertices = vertices.shape[0]
	mask = np.all((faces >= 0) & (faces < n_vertices), axis=1)
	faces = faces[mask]
	
	pymesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
	ms = pymeshlab.MeshSet()
	ms.add_mesh(pymesh)
	ms.meshing_remove_connected_component_by_face_number(mincomponentsize=1000)
	ms.apply_coord_taubin_smoothing()
	print("Computing normals in pymeshlab")
	ms.compute_normal_per_vertex(weightmode=0)
	
	ml_mesh = ms.current_mesh()
	vertices = np.array(ml_mesh.vertex_matrix(), dtype=np.float64)
	faces = np.array(ml_mesh.face_matrix(), dtype=np.int32)
	normals = np.array(ml_mesh.vertex_normal_matrix(), dtype=np.float64)
	
	o3d_mesh = o3d.geometry.TriangleMesh()
	o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
	o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
	o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
	o3d_mesh.remove_unreferenced_vertices()
	o3d_mesh.remove_degenerate_triangles()
	o3d_mesh.remove_non_manifold_edges()
	o3d_mesh.remove_duplicated_vertices()
	o3d_mesh.remove_duplicated_triangles()
	
	return o3d_mesh
	

def generate_source_pc(target_p, xyz, label, target_points, output_dir, outlier_dir):
    target_pc = o3d.geometry.PointCloud()
    target_pc.points = o3d.utility.Vector3dVector(target_p)
    
    print(xyz)
    source = o3d.io.read_triangle_mesh(xyz)
    source.remove_unreferenced_vertices()
    source.remove_degenerate_triangles()
    source.remove_non_manifold_edges()
    source.remove_duplicated_vertices()
    source.remove_duplicated_triangles()
    
    verts = np.asarray(source.vertices)
    if np.isnan(verts).any() or np.isinf(verts).any():
        print("Mesh has invalid vertex values!")
    
    source = get_normals(source)
    print("Computed Normals using original Mesh")
    
    source_pc = o3d.geometry.PointCloud()
    source_p = source.vertices
    source_norms = source.vertex_normals
    source_pc.points = o3d.utility.Vector3dVector(source_p)
    source_pc.normals = o3d.utility.Vector3dVector(source_norms)
    
    source_translation = source_pc.get_center()
    source_pc.translate(-source_translation)
    source.translate(-source_translation)
    
    new_source_verts, source_scale = normalize_points(source_pc)
    
    if len(source_pc.points) > target_points:
        ms_target = pymeshlab.MeshSet()
        mesh = pymeshlab.Mesh(vertex_matrix=np.asarray(source_pc.points),
                              face_matrix=np.asarray(source.triangles),
                              v_normals_matrix=np.asarray(source_pc.normals))
        ms_target.add_mesh(mesh)
        ms_target.generate_sampling_poisson_disk(samplenum = int(target_points + 50), exactnumflag=True)
        new_mesh = ms_target.current_mesh()
        new_points = new_mesh.vertex_matrix()
        down_norms = new_mesh.vertex_normal_matrix()
        source_pc.points = o3d.utility.Vector3dVector(new_points)
    
        source_pc = source_pc.farthest_point_down_sample(target_points)
        down_norms = np.asarray(source_pc.normals)
    else:
        return None, None
    
    
    voxel_size = 1.0
    distance_threshold = voxel_size * 5
    
    print("[Progress] Alignment", flush = True)
    start1 = time.time()
    
    source_pc, s_source = normalize_points(source_pc)
    
    end1 = time.time()
    
    print_elapsed(start1, end1)
    
    voxel_size = 1.0
    
    print("[Progress] ICP")
    start2 = time.time()
    
    distance_threshold = voxel_size * 30
    
    icp_result = o3d.pipelines.registration.registration_icp(
    source_pc, target_pc, distance_threshold, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    
    source.vertices = o3d.utility.Vector3dVector(np.asarray(new_source_verts.points))
    source = source.transform(icp_result.transformation)
    
    source_path = save_reg(source, xyz, label, output_dir)
    
    end2 = time.time()
    
    print_elapsed(start2, end2)
    
    return source_path, source_scale
				
def save_reg(pc, xyz, label, output_dir):
	base_name = os.path.splitext(os.path.basename(xyz))[0]
	out_path1 = os.path.join(output_dir, f"vertebra{label}", f"{base_name}_reg.ply")
	os.makedirs(os.path.join(output_dir, f"vertebra{label}"), exist_ok=True)
	o3d.io.write_triangle_mesh(out_path1, pc)
	
	return out_path1
	
def save_reg_mean(mean_pc, label, mean_label, num, output_dir2):
	out_path2 = os.path.join(output_dir2, f"vertebra{label}", f"vertebra{label}_mean{mean_label}_n_{num}.ply")
	os.makedirs(os.path.join(output_dir2, f"vertebra{label}"), exist_ok=True)
	o3d.io.write_triangle_mesh(out_path2, mean_pc)


def single_label_pc(input_xyz, output_dir, output_dir2, label, outlier_dir):
    print('[Progress] Cleaning inputs')    
    
    # n_jobs = 10
    input_xyz = Parallel(n_jobs=10)(delayed(clean_inputs)(xyz, label) for xyz in input_xyz)
    input_xyz = [r for r in input_xyz if r is not None]
    n = len(input_xyz)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)
    os.makedirs("outlier_pcs", exist_ok=True)
    
    if n != 0:
        os.makedirs(os.path.join(output_dir, f"vertebra{label}"), exist_ok = True)
    
    # set target and save target point cloud
    print('[Progress] Reading in Target')

    print(input_xyz[-1])
    
    target = o3d.io.read_triangle_mesh(input_xyz[-1])
    target.remove_unreferenced_vertices()
    target.remove_degenerate_triangles()
    target.remove_non_manifold_edges()
    target.remove_duplicated_vertices()
    target.remove_duplicated_triangles()
    
    target = get_normals(target)
    
    target_p = target.vertices
    target_norms = target.vertex_normals
    print(np.asarray(target_norms).shape)
    target_pc = o3d.geometry.PointCloud()
    target_pc.points = o3d.utility.Vector3dVector(np.asarray(target_p))
    target_pc.normals = o3d.utility.Vector3dVector(np.asarray(target_norms))
    target_translation = target_pc.get_center()
    target_pc.translate(-target_translation)
    target.translate(-target_translation)
    
    new_target_verts, target_scale = normalize_points(target_pc)
    
    if label in np.arange(1.0, 51.0, 1.0):
        target_points = 1000
    else:
        target_points = 1000
    
    if len(target_pc.points) > target_points:
        ms_target = pymeshlab.MeshSet()
        mesh = pymeshlab.Mesh(vertex_matrix=np.asarray(target_pc.points),
                              face_matrix=np.asarray(target.triangles),
                              v_normals_matrix=np.asarray(target_pc.normals))
        ms_target.add_mesh(mesh)
        ms_target.generate_sampling_poisson_disk(samplenum=int(target_points + 50), exactnumflag=True)
        new_mesh = ms_target.current_mesh()
        new_points = new_mesh.vertex_matrix()
        down_norms = new_mesh.vertex_normal_matrix()
        target_pc.points = o3d.utility.Vector3dVector(new_points)
    
        target_pc= target_pc.farthest_point_down_sample(target_points)
        down_norms = np.asarray(target_pc.normals)
    
    print(type(target_pc))
    
    print(target_pc)
    
    target_pc, s_target = normalize_points(target_pc)
    
    target_pc.normals = o3d.utility.Vector3dVector(down_norms)
    target_p = np.asarray(target_pc.points)
    target_norms = np.asarray(target_pc.normals)
    target.vertices = o3d.utility.Vector3dVector(np.asarray(new_target_verts.points))
    
    print(np.asarray(target_pc.points).shape)
    save_reg(target, input_xyz[-1], label, output_dir)
    i = 1
    

    target_file = [input_xyz[-1]]    
    input_xyz = input_xyz[:-1]

    params = [(target_p, xyz, label, target_points, output_dir, outlier_dir)
        for j, xyz in enumerate(input_xyz)]
    
    # n_jobs = 6
    results = Parallel(n_jobs=5)(delayed(generate_source_pc)(*p) for p in params)
    filenames, scales = zip(*results)
    filenames = [r for r in filenames if r is not None]
    filenames = filenames + target_file
    scales = [r for r in scales if r is not None]
    scales = np.asarray(scales + [s_target])
    
    out_folder = f"pca_rsna/vertebra_{label}"
    os.makedirs(out_folder, exist_ok=True)
    np.savez(os.path.join(out_folder, "scales.npz"), filenames = filenames, scales = scales)


    # Perform Hungarian/JV algorithem
    # results = Parallel(n_jobs=6)(delayed(hungarian)(filename, target_p) for filename in filenames)
    # ps, norms = zip(*results)
    # ps = [r for r in ps if r is not None]
    # ps = ps + [target_p]
    
    # norms = [r for r in norms if r is not None]
    # norms = norms + [target_norms]
    
    # ps = np.stack(ps)
    # norms = np.stack(norms)
    # mean_all_points = ps.mean(axis = 0)
    # mean_all_norms = norms.mean(axis = 0)
    # mean_all = o3d.geometry.PointCloud()
    # mean_all.points = o3d.utility.Vector3dVector(mean_all_points)
    # mean_all.normals = o3d.utility.Vector3dVector(mean_all_norms)
    
    # save_reg_mean(mean_all, label, 0, len(ps), output_dir2)

def batch_process_folder(folder_path, output_dir, output_dir2, outlier_dir):
	start = time.time()
	
	#labels = np.arange(1.0, 101.0, 1.0)
    # 40s are lumbar vertebrae
    # 90s and 100 are lumbar discs
    # rest are spinal cord segments associated with discs
	labels = [
		41,
		# 91,
		# 291,
		# 42,
		# 43, 44, 45,
		# 92, 93, 94, 95, 100,
		# 292, 293,294,295, 2100
		]

	for label in labels:
		try:
			suffix = f"**/*_vertebra{label}.ply"
			

			xyz_files = sorted(glob.glob(os.path.join(folder_path, suffix), recursive=True))
			
			print(f"Starting registration for label {label}.")
			start_chunk = time.time()
			
			if len(xyz_files) == 0:
				print("No files found")
				continue
			
			single_label_pc(xyz_files, output_dir, output_dir2, label, outlier_dir)
			
			end_chunk = time.time()
			
			print(f"[Progress] Finished registration for label {label}.")
			print_elapsed(start_chunk, end_chunk)
			
		except Exception as e:
				print(f"Failed for label {label} found:", e)
				raise

	end = time.time()

	print(f"Complete.")
	print_elapsed(start, end)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Point Cloud Registration")
	parser.add_argument("input", help="Directory containing point mesh files")
	parser.add_argument("--r", type=str, default="reg_pointclouds", help="Directory to save registered outputs")
	parser.add_argument("--m", type=str, default="mean_pointclouds", help="Directory to save mean outputs")
	parser.add_argument("--o", type=str, default="outlier_pcs", help="Directory to save outliers")
	args = parser.parse_args()

	if os.path.isdir(args.input):
		batch_process_folder(args.input, args.r, args.m, args.o)
	else:
		print(f"[ERROR] Invalid input path: {args.input}")
