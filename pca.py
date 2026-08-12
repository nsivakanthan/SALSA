import os
import argparse
import glob
import time
from joblib import Parallel, delayed
import re
import math
import tempfile

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import trimesh
import plotly.graph_objects as go
from moviepy import ImageSequenceClip
from scipy.optimize import linear_sum_assignment
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pymeshlab
import matplotlib.cm as cm
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import pyvista as pv
import random
import igl
import matplotlib as mpl
from PIL import Image, ImageDraw, ImageFont
from matplotlib.colors import LinearSegmentedColormap

# delete mean.npy to trigger redo of SVD
clean = True
# scale mesh to true size
scale = False

def print_elapsed(start, end):
	elapsed = end - start
	
	hours = int(elapsed // 3600)
	
	minutes = int((elapsed % 3600) // 60)
	
	seconds = int(elapsed % 60)
	
	print(f"[Progress] Elapsed time: {hours}h {minutes}m {seconds}s")

def get_filenames(label, file_path = "reg_pointclouds"):
	folder_path = os.path.join(os.getcwd(), file_path)

	suffix = f"**/*vertebra{label}*.ply"

	ply_files = sorted(glob.glob(os.path.join(folder_path, suffix), recursive=True))

	return ply_files

def fix_frame(filename, target_height, target_width):
	frame = np.array(Image.open(filename))
	h, w, _ = frame.shape

	# Crop or pad height
	if h > target_height:
		frame = frame[:target_height, :, :]
	elif h < target_height:
		pad = target_height - h
		frame = np.pad(frame, ((0,pad),(0,0),(0,0)), mode='edge')

	# Crop or pad width
	if w > target_width:
		frame = frame[:, :target_width, :]
	elif w < target_width:
		pad = target_width - w
		frame = np.pad(frame, ((0,0),(0,pad),(0,0)), mode='edge')

	# Ensure both dimensions divisible by 2
	frame = frame[:frame.shape[0]//2*2, :frame.shape[1]//2*2, :]
	return frame

	
def elbow(centered, S, output_dir, label, z):
	# Create Elbow plot for PCA

	var_explained = (S ** 2)/ (centered.shape[0] - 1)
	# var_explained = S
	var_ratio = var_explained /np.sum(var_explained)
	cum_var = np.cumsum(var_ratio)
	num_comp = min(centered.shape[0], centered.shape[1])
	n_comp_50 = np.argmax(cum_var >= .5) + 1
	n_comp_70 = np.argmax(cum_var >= .7) + 1
	n_comp_90 = np.argmax(cum_var >= .9) + 1

	if z ==2:
		num_comp = min(centered.shape[0], centered.shape[1]-1)
	
	plt.figure(figsize = (8,6))
	plt.plot(np.arange(1, num_comp+1),cum_var[:num_comp] * 100,color = 'blue')
	plt.hlines(y=90, xmin=0, xmax=n_comp_90, color='green', linestyle='--')
	plt.vlines(x=n_comp_90, ymin =0, ymax=90, color='green', linestyle='--', label=f'{n_comp_90} components')
	plt.hlines(y=70, xmin=0, xmax=n_comp_70, color='r', linestyle='--')
	plt.vlines(x=n_comp_70, ymin=0, ymax=70, color='r', linestyle='--', label=f'{n_comp_70} components')
	plt.hlines(y=50, xmin=0, xmax=n_comp_50, color='orange', linestyle='--')
	plt.vlines(x=n_comp_50, ymin=0, ymax=50, color='orange', linestyle='--', label=f'{n_comp_50} components')
	plt.xlabel('Number of Principal Components')
	plt.xticks(np.arange(0, math.ceil(num_comp/50.0)*50 + 50, 50))
	plt.ylabel('Cumulative Variance Explained (%)')
	plt.yticks(np.arange(0,110,10))
	plt.title(f'Cumulative Variance Explained by Principal Components (Total: {num_comp})')
	plt.tight_layout()
	plt.legend(loc='lower right')

	filepath = os.path.join(output_dir, f"vertebra_{label}")
	os.makedirs(filepath, exist_ok = True)
	filename = f"variance_explained_vertebra_{label}_{z}.png"
	plt.savefig(os.path.join(filepath, filename), dpi = 300)

def axis_plot(centered, U, S, Vt, label, output_dir, plys, axes, z):
	# Project shapes onto 3 PC and get figure

	# provide list of integers for principal directions Ex. [0,1,2]
	pc_projection = centered @ Vt[axes,:].T
	
	center = np.mean(pc_projection, axis = 0)
	distances = np.linalg.norm(pc_projection-center, axis =1)
	threshold = np.percentile(distances, 98.5)
	outlier_indices = np.where(distances > threshold)[0]
	files_list = [plys[i] for i in outlier_indices]
	print(files_list)
	print(" ".join(f'"{os.path.basename(p)}"' for p in files_list))

	fig = go.Figure()
	fig.add_trace(go.Scatter3d(
	x=pc_projection[:, 0],
	y=pc_projection[:, 1],
	z=pc_projection[:, 2],
	mode='markers',
	marker=dict(
		size=5,
		color='blue',
		opacity=0.8
	)
	))

	fig.update_layout(
	scene=dict(
		xaxis_title=f'PC{axes[0] +1}',
		yaxis_title=f'PC{axes[1] +1}',
		zaxis_title=f'PC{axes[2] +1}'
	),
	title=''
	)

	filepath = os.path.join(output_dir, f"vertebra_{label}")
	os.makedirs(filepath, exist_ok = True)
	filename = f"projection_plot_vertebra_{label}_{z}.png"

	fig.write_image(os.path.join(filepath, filename), width = 2000, height = 1400)


def get_pca_meshes(pc, transf, label):
	# needed for reconstructing surface from point clouds (JV registration)
    # pc.apply_translation(transf)
    
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(vertex_matrix=np.asarray(pc.vertices), v_normals_matrix=np.asarray(pc.vertex_normals))
    ms.add_mesh(mesh)
    if label in [41,42,43,44,45,"vertebra"]:
        ms.generate_marching_cubes_rimls(resolution = 50)
        ms.meshing_re_orient_faces_by_geometry()
        ms.generate_surface_reconstruction_screened_poisson(depth=8
                                                            # , samplespernode=0.2
                                                            ,preclean=True)
        ms.generate_resampled_uniform_mesh()
    else:
        ms.generate_surface_reconstruction_screened_poisson(depth=8,preclean=True)
    m = ms.current_mesh()
    vertices = m.vertex_matrix()
    faces = m.face_matrix()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.translate(transf)
    
    ref_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc.vertices))
    kdt = o3d.geometry.KDTreeFlann(ref_pc)
    
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles), kdt

def get_meshes(pc, transf, label):
    # pc.apply_translation(transf)
    print(np.asarray(pc.vertices).shape, flush = True)
    print(np.asarray(pc.vertex_normals).shape, flush = True)
    
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(vertex_matrix=np.asarray(pc.vertices), v_normals_matrix=np.asarray(pc.vertex_normals))
    ms.add_mesh(mesh)

    if label in [41,42,43,44,45,"vertebra"]:
        ms.generate_marching_cubes_rimls(resolution = 50)
        ms.meshing_re_orient_faces_by_geometry()
        ms.generate_surface_reconstruction_screened_poisson(depth=8
                                                            # , samplespernode=0.2
                                                            ,preclean=True)
        ms.generate_resampled_uniform_mesh()
    else:
        m = ms.current_mesh()
        print("Vertices:", m.vertex_number(), "Faces:", m.face_number(), flush = True)
        ms.compute_normal_for_point_clouds()
        ms.generate_surface_reconstruction_screened_poisson(depth=8,preclean=True)

    tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False).name
    ms.save_current_mesh(tmp)
    m = ms.current_mesh()
    print("Vertices:", m.vertex_number(), "Faces:", m.face_number(), flush = True)
    mesh = o3d.io.read_triangle_mesh(tmp)
    os.remove(tmp)
    
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)


def generate_mean_frame_open3d(rotation_axis, rotation_angle, transfs, label, k, output_dir, displacements_list, meshes_vertices,
		 meshes_triangles, disp_min, disp_max, cmap):
	# helper for mean figure

	meshes = []
	for (vertices, triangles,transf) in zip(meshes_vertices, meshes_triangles, transfs):
		vertices = np.array(vertices, dtype=np.float64, order="C", copy=True)
		triangles = np.array(triangles, dtype=np.int32, order="C", copy=True)
		mesh = o3d.geometry.TriangleMesh()
		mesh.vertices = o3d.utility.Vector3dVector(vertices)
		mesh.triangles = o3d.utility.Vector3iVector(triangles)
		mesh.compute_vertex_normals()
		mesh.translate(transf)
		meshes.append(mesh)

	R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_angle * np.array(rotation_axis))

	vis = o3d.visualization.Visualizer()
	vis.create_window(visible=False, width=1920, height=1080)
	for mesh, displacement in zip(meshes, displacements_list):
		# disp_norm = (displacement - disp_min) / (disp_max - disp_min + 1e-8)
		disp_norm = displacement / (max(abs(disp_min), abs(disp_max)) + 1e-8)
		disp_norm = (disp_norm + 1) / 2
		# print(disp_norm)
		colors = cmap(disp_norm)[:, :3]
		mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
		vis.add_geometry(mesh)
		mesh.rotate(R, center=mesh.get_center())
		vis.update_geometry(mesh)

	render_option = vis.get_render_option()
	render_option.mesh_show_back_face = True
	render_option.mesh_show_wireframe = True
	render_option.mesh_shade_option = o3d.visualization.MeshShadeOption.Color
	render_option.light_on = False

	ctr = vis.get_view_control()

	for _ in range(20):  # decrease FOV repeatedly
		ctr.change_field_of_view(step=-1.0)
	ctr.set_up([0, 1, 0])
	ctr.set_zoom(.33)
	vis.poll_events()
	vis.update_renderer()

	path = os.path.join(output_dir, f"vertebra_{label}", "images", f"mean")
	os.makedirs(path, exist_ok=True)
	name = f"vertebra{label}_mean_{k}.png"
	filename = os.path.join(path, name)

	vis.capture_screen_image(filename, do_render=True)
	root, _ = os.path.splitext(filename)
	if k % 3 == 0:
		vis.capture_screen_image(root + '.jpg', do_render=True)
	vis.destroy_window()

	return filename

def vis_mean_o3d(centered, mean, norms, label):
	# Create images and videos of mean with a sample of shapes

	cmap = cm.get_cmap("RdYlBu")

	# np.random.seed(96)
	num_frames = 120
	num_samples = 6
	indices = np.random.choice(centered.shape[0], num_samples, replace = False)
	samples = centered[indices]
	sample_norms = norms[indices]

	axes = np.array([[0, 1, 0], [1, 0, 0]])
	rotation_axes = np.repeat(axes, [num_frames / 2, num_frames / 2], axis=0)

	rotation_angles = np.arange(0, 2 * np.pi, 2 * np.pi / (num_frames / 2))
	rotation_angles = np.concatenate((rotation_angles, rotation_angles))
	fps = 10

	i = 0
	j = 0
	pcs = []
	transfs = []
	for sample, sample_norm in zip(samples, sample_norms):
		pc = trimesh.Trimesh(vertices = sample.reshape(-1, 3))
		pc.vertex_normals = sample_norm.reshape(-1, 3)
		pcs.append(pc)

		# translations depend on if meshes are scaled back to original size or not
		# transf = np.asarray([3.5 * i, 3.5 * j, 0])
		transf = np.asarray([90 * j, 90 * i, 0])
		i = i + 1
		if i > 1:
			i = 0
			j = j + 1
		transfs.append(transf)

	mean_pc = trimesh.Trimesh(vertices = mean.reshape(-1,3))
	mean_norms = norms.mean(axis = 0).reshape(-1,3)
	mean_pc.vertex_normals = mean_norms.reshape(-1, 3)

	# mean_transf = np.asarray([1.75, 3.5 * j, 0])
	mean_transf = np.asarray([90 * j, 45, 0])

	mean_verts, mean_tris, _ = get_pca_meshes(mean_pc, np.asarray([0, 0, 0]), label)
	transfs.append(mean_transf)

	results = Parallel(n_jobs=2)(delayed(get_meshes)(pc, transf, label) for pc, transf in zip(pcs, transfs[:-1]))

	meshes_vertices, meshes_triangles = zip(*results)

	meshes_vertices = list(meshes_vertices)
	meshes_triangles = list(meshes_triangles)

	meshes_vertices.append(mean_verts)
	meshes_triangles.append(mean_tris)

	mean_vertices = meshes_vertices[-1]
	mean_triangles = meshes_triangles[-1]

	params = [
		(k, mesh_vertices, mesh_triangles, mean_vertices, mean_triangles, transf, transfs[-1])
		for k, (mesh_vertices, mesh_triangles, transf) in
		enumerate(zip(meshes_vertices[:-1], meshes_triangles[:-1], transfs[:-1]))
	]
	displacements_list = Parallel(n_jobs=2)(delayed(displacements)(*p) for p in params)

	all_displacements = np.concatenate(displacements_list)
	disp_min = all_displacements.min()
	disp_max = all_displacements.max()

	displacements_list.append(np.zeros(len(mean_vertices)))

	params = [
		(rotation_axis, rotation_angle, transfs, label, k, output_dir, displacements_list, meshes_vertices,
		 meshes_triangles, disp_min, disp_max, cmap)
		for k, (rotation_angle, rotation_axis) in enumerate(zip(rotation_angles, rotation_axes))
	]
	filenames = Parallel(n_jobs=3)(delayed(generate_mean_frame_open3d)(*p) for p in params)

	path = os.path.join(output_dir, f"vertebra_{label}", "videos")
	os.makedirs(path, exist_ok=True)

	name = f"vertebra{label}_mean.mp4"
	filename = os.path.join(path, name)

	frames_fixed = [fix_frame(f,1080,1920) for f in filenames]

	clip = ImageSequenceClip(frames_fixed, fps=fps)
	clip.write_videofile(filename, codec="libx264",
						 audio=False,  # no audio stream
						 ffmpeg_params=["-pix_fmt", "yuv420p", "-profile:v", "baseline"])

	for img in filenames:
		os.remove(img)
	
def magnitude_color(mean, pd, scaling):
	#mean_points = mean.reshape(N,3)
	#mean_center = np.mean(mean_points, axis = 0)
	pc_dir = pd.reshape(-1,3)
	
	magnitude = np.linalg.norm(pc_dir * scaling, axis =1)
		
	return magnitude
	
def normalize_magnitudes(magnitudes):
	magnitudes = np.array(magnitudes)
	if np.allclose(0, magnitudes.max()):
		magnitudes_norm = np.zeros_like(magnitudes)
	else:
		magnitudes_norm = magnitudes/magnitudes.max()
	
	return magnitudes_norm

def generate_frame_open3d(rotation_axis, rotation_angle, transfs, label, component, k, output_dir, displacements, meshes_vertices, meshes_triangles, disp_min, disp_max, cmap):
	# helper for PCA directions
	
	meshes = []
	for (vertices, triangles, transf) in zip(meshes_vertices, meshes_triangles, transfs):
		mesh = o3d.geometry.TriangleMesh()
		mesh.vertices = o3d.utility.Vector3dVector(vertices)
		mesh.triangles = o3d.utility.Vector3iVector(triangles)
		mesh.compute_vertex_normals()
		# print(transf)
		mesh.translate(transf)
		meshes.append(mesh)
	# print(len(meshes))
	R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_angle * np.array(rotation_axis))

	vis = o3d.visualization.Visualizer()
	vis.create_window(visible = False, width=1920, height=1080)


	for mesh, displacement in zip(meshes, displacements):
		# disp_norm = (displacement - disp_min) / (disp_max - disp_min + 1e-8)
		disp_norm = displacement / (max(abs(disp_min), abs(disp_max)) + 1e-8)
		disp_norm = (disp_norm +1)/2
		# print(disp_norm)
		colors = cmap(disp_norm)[:, :3]
		mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
		vis.add_geometry(mesh)
		mesh.rotate(R, center=mesh.get_center())
		vis.update_geometry(mesh)

	render_option = vis.get_render_option()
	render_option.mesh_show_back_face = True
	render_option.mesh_show_wireframe = True
	render_option.mesh_shade_option = o3d.visualization.MeshShadeOption.Color
	render_option.light_on = False

	ctr = vis.get_view_control()
	for _ in range(20):  # decrease FOV repeatedly
		ctr.change_field_of_view(step=-1.0)
	ctr.set_up([1,0,0])
	ctr.set_zoom(.33)
	vis.poll_events()
	vis.update_renderer()

	path = os.path.join(output_dir, f"vertebra_{label}", "images", f"direction{component + 1}")
	os.makedirs(path, exist_ok=True)
	name = f"vertebra{label}_direction{component + 1}_{k}.png"
	filename = os.path.join(path, name)

	vis.capture_screen_image(filename, do_render = True)
	root, _ = os.path.splitext(filename)
	if k % 10 == 0:
		vis.capture_screen_image(root + '.jpg', do_render = True)
	vis.destroy_window()

	return filename
	
def displacements(i, mesh_vertices, mesh_triangles, mean_vertices, mean_triangles, transf, mean_transf):
    # mesh_ref = trimesh.Trimesh(vertices=np.asarray(mean_vertices), faces=np.asarray(mean_triangles))
    # mesh_query = trimesh.Trimesh(vertices=np.asarray(mesh_vertices), faces=np.asarray(mesh_triangles))
    
    # mesh_ref.apply_translation(-mean_transf)
    # mesh_query.apply_translation(-transf)
    
    # mesh_ref.fix_normals()
    # mesh_query.fix_normals()
    # displacement = trimesh.proximity.signed_distance(mesh_ref, mesh_query.vertices)
	displacement = np.linalg.norm(mesh_vertices - mean_vertices, axis=1)
    # print(displacement)
	return displacement




def vis_principal_directions_o3d(Vt, S, mean, norms, label):
	# Create images and videos of pca directions displayed side by side at different scales for each PC
	cmap = cm.get_cmap("RdYlBu")

	components = np.arange(0, 3)
	scalings = [-3, 3, -2, 2, -1, 1, 0]

	num_frames = 120

	axes = np.array([[0, 1, 0], [1, 0, 0]])
	rotation_axes = np.repeat(axes, num_frames // 2, axis=0)

	rotation_angles = np.linspace(0, 2 * np.pi, num_frames // 2, endpoint=False)
	rotation_angles = np.concatenate((rotation_angles, rotation_angles))
	fps = 10

	mean_pc = trimesh.Trimesh(vertices = mean.reshape(-1, 3))
	mean_norms = norms.mean(axis=0).reshape(-1, 3)
	mean_pc.vertex_normals = mean_norms.reshape(-1, 3)

	mean_vertices, mean_triangles, kdt = get_pca_meshes(mean_pc, np.asarray([0,0,0]), label)

	for component in components:
		i = 0
		j = 0
		meshes_vertices = []
		meshes_triangles = []
		transfs = []
		for scaling in scalings:
			pd = Vt[component].reshape(-1, 3)

			new_pc = np.empty_like(mean_vertices)

			print("Doing KNN search")
			for n, p in enumerate(mean_vertices):
				cnt, idx, dist = kdt.search_knn_vector_3d(p, 5)
				weights = 1.0 / (np.sqrt(dist) + 1e-8)
				weights /= weights.sum()
				# m = idx[0]
				direction = np.sum(weights[:,None] * pd[idx], axis=0)
				new_pc[n] = p + scaling * direction
			print("Completed KNN search")
			# new_pc = mean_vertices + (np.sqrt(S[component])/(len(S)-1) * scaling * pd).reshape(-1, 3)
			new_p = new_pc
			print(new_p.shape)

			# pc = trimesh.points.PointCloud(new_p)

			if scaling == 0:
				# transf = np.asarray([1.75, 3.5 * j, 0])
				# transf = np.asarray([1, 1.5 * j, 0])
				transf = np.asarray([30, 80 * j, 0])
			else:
				# transf = np.asarray([3.5 * i, 3.5 * j, 0])
				# transf = np.asarray([1.5 * i, 1.5 * j, 0])
				transf = np.asarray([60 * i, 80 * j, 0])
			i = i + 1
			if i >1:
				i = 0
				j = j + 1
			transfs.append(transf)
			meshes_vertices.append(new_p)
			meshes_triangles.append(mean_triangles)
		print(len(meshes_vertices), len(meshes_triangles), len(transfs))
		print(transfs)

		# results = Parallel(n_jobs = 2)(delayed(get_pca_meshes)(pc, transf, label) for pc, transf in zip(pcs, transfs))

		# meshes_vertices, meshes_triangles = zip(*results)
		
		mean_verts = meshes_vertices[-1]
		mean_tris = meshes_triangles[-1]

		params = [
			(k, mesh_vertices, mesh_triangles, mean_verts, mean_tris, transf, transfs[-1])
			for k, (mesh_vertices, mesh_triangles, transf) in enumerate(zip(meshes_vertices[:-1], meshes_triangles[:-1], transfs[:-1]))
		]
		displacements_list = Parallel(n_jobs=2)(delayed(displacements)(*p) for p in params)


		all_displacements = np.concatenate(displacements_list)
		disp_min = all_displacements.min()
		disp_max = all_displacements.max()

		displacements_list.append(np.zeros(len(mean_verts)))
		print(len(displacements_list))

		params = [
			(rotation_axis, rotation_angle, transfs, label, component, k, output_dir, displacements_list, meshes_vertices, meshes_triangles, disp_min, disp_max, cmap)
			for k, (rotation_angle, rotation_axis) in enumerate(zip(rotation_angles, rotation_axes))
		]
		filenames = Parallel(n_jobs=3)(delayed(generate_frame_open3d)(*p) for p in params)

		path = os.path.join(output_dir, f"vertebra_{label}", "videos")
		os.makedirs(path, exist_ok = True)

		name = f"vertebra{label}_direction{component + 1}.mp4"
		filename = os.path.join(path, name)

		frames_fixed = [fix_frame(f,1080,1920) for f in filenames]

		clip = ImageSequenceClip(frames_fixed, fps=fps)
		clip.write_videofile(filename, codec="libx264",
							 audio=False,  # no audio stream
							 ffmpeg_params=["-pix_fmt", "yuv420p", "-profile:v", "baseline"])

		for img in filenames:
			os.remove(img)

def generate_frame_open3d_scale(rotation_axis, rotation_angle, transf, label, component, k, output_dir, displacement, mesh_vertices, mesh_triangles, disp_min, disp_max, cmap):
	# helper for PCA scale
	R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_angle * np.array(rotation_axis))
	
	mesh = o3d.geometry.TriangleMesh()
	mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
	mesh.triangles = o3d.utility.Vector3iVector(mesh_triangles)
	mesh.compute_vertex_normals()
	mesh.rotate(R, center=mesh.get_center())
	mesh.translate(transf)

	disp_norm = displacement / (disp_max + 1e-8)
	# print(disp_norm)
	cmap=cmap.reversed()
	colors = cmap(disp_norm)[:, :3]
	mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

	plotter = pv.Plotter(off_screen=True, window_size=(int(1920/2), 1080))
	
	verts = np.asarray(mesh.vertices)
	faces = np.asarray(mesh.triangles)

	faces_pv = np.hstack([
		np.full((len(faces),1), 3),
		faces
	]).astype(np.int64).flatten()

	pv_mesh = pv.PolyData(verts, faces_pv)

	pv_mesh["colors"] = np.asarray(mesh.vertex_colors)

	actor = plotter.add_mesh(
		pv_mesh,
		scalars="colors",
		rgb=True,
		show_edges=True,
		edge_color="black"
	)
	actor.GetProperty().SetEdgeOpacity(0.5)

	plotter.camera_position = "iso"
	# plotter.view_isometric()
	# plotter.camera.parallel_projection = True
	plotter.view_xy()
	plotter.camera.zoom(1.3)

	path = os.path.join(output_dir, f"vertebra_{label}", "images", f"direction{component + 1}_scale")
	os.makedirs(path, exist_ok=True)
	name = f"vertebra{label}_direction{component + 1}_{k}.png"
	filename = os.path.join(path, name)

	plotter.show(screenshot=filename)

	root, _ = os.path.splitext(filename)

	# ============================
	# Add matplotlib colorbar
	# ============================
	img = Image.open(filename).convert("RGB")

	draw = ImageDraw.Draw(img)

	title = f"Principal Deformation {component+1}"

	mesh_w, mesh_h = img.size

	font = ImageFont.truetype(
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    60
	)

	bbox = draw.textbbox((0,0), title, font=font)
	text_width = bbox[2] - bbox[0]

	x = (img.width - text_width) // 2
	y = 30

	draw.text(
    (x,y),
    title,
    font=font,
    fill=(0,0,0))


	# Generate colorbar
	# fig, ax = plt.subplots(figsize=(0.6, 3.5))
	fig, ax = plt.subplots(figsize=(3.5, .6))

	norm = mpl.colors.Normalize(
		# vmin=-max_abs,
		# vmax=max_abs
		vmin = disp_min,
		vmax = disp_max
	)

	sm = mpl.cm.ScalarMappable(
		cmap=cmap,
		norm=norm
	)
	sm.set_array([])

	cbar = fig.colorbar(sm, ax=ax, orientation="horizontal")
	cbar.set_label("")
	cbar.ax.set_title("Distance from Mean",fontsize=8,pad=8)

	# make tick labels smaller
	cbar.ax.tick_params(labelsize=7)

	ax.remove()

	cbar_file = filename.replace(".png", "_cbar.png")

	plt.savefig(
		cbar_file,
		dpi=300,
		bbox_inches="tight",
		pad_inches=0.05,
		facecolor="white"
	)

	plt.close(fig)
	# Load colorbar
	cbar_img = Image.open(cbar_file).convert("RGB")

	# Scale colorbar relative to mesh image
	# target_height = int(mesh_w * 0.5)   # 35% of image height

	# ratio = target_height / cbar_img.height

	# cbar_img = cbar_img.resize(
	# 	(
	# 		int(cbar_img.width * ratio),
	# 		target_height
	# 	)
	# )
	target_width = int(mesh_w * 0.35)

	ratio = target_width / cbar_img.width
	cbar_img = cbar_img.resize((
		target_width,
		int(cbar_img.height * ratio)
	))


	# Put it inside the image
	# margin_right = int(mesh_w/2)
	# margin_top = int(mesh_h *8/10)

	# position = (mesh_w - cbar_img.width - margin_right,margin_top)
	margin_bottom = 0

	position = (																																																																																												
		(mesh_w - cbar_img.width) // 2,
		mesh_h - cbar_img.height - margin_bottom
	)																						

	img.paste(cbar_img,position)


	# Save back
	img.save(filename)

	os.remove(cbar_file)

	# jpg output
	root, _ = os.path.splitext(filename)
	if k % 5 == 0:
		img = Image.open(filename)
		img.convert("RGB").save(root + ".jpg")

	return filename

def vis_principal_directions_o3d_scale(Vt, S, mean, mean_tris, norms, label):
	# Create images and videos of mean shape deforming over time in PC directions

	cmap = cm.get_cmap("RdYlBu")
	components = np.arange(0, 3)

	num_frames = 120

	norm_coef = 2
	segment1 = np.linspace(0, norm_coef, num_frames//4, endpoint=False)
	segment2 = np.linspace(norm_coef, 0, num_frames//4, endpoint=False)
	segment3 = np.linspace(0, -norm_coef, num_frames//4, endpoint=False)
	segment4 = np.linspace(-norm_coef, 0, num_frames//4)

	scalings = np.concatenate([segment1, segment2, segment3, segment4])

	axes = np.array([[0, 1, 0], [1, 0, 0]])
	rotation_axes = np.repeat(axes, num_frames // 2, axis=0)

	rotation_angles = np.linspace(0, 2 * np.pi, num_frames // 2, endpoint=False)
	rotation_angles = np.concatenate((rotation_angles, rotation_angles))

	# rotation_axis = np.array([0, 1, 0])
	# rotation_angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)
	# rotation_axes = np.repeat(rotation_axis[np.newaxis, :], num_frames, axis=0)

	fps = 10

	mean_pc = trimesh.Trimesh(vertices=mean.reshape(-1, 3))
	mean_norms = norms.mean(axis=0).reshape(-1, 3)
	mean_pc.vertex_normals = mean_norms.reshape(-1, 3)

	mean_vertices = mean_pc.vertices
	mean_triangles = mean_tris

	transf = np.asarray([0, 0, 0])

	for component in components:
		i = 0
		j = 0
		meshes_vertices = []
		for scaling in scalings:
			scaling = scaling * S[component]/np.sqrt(len(S)-1)
			pd = Vt[component].reshape(-1, 3)

			new_pc = mean_vertices + pd
			meshes_vertices.append(new_pc)

		mean_verts = mean_vertices.reshape(-1, 3)
		mean_tris = mean_triangles.reshape(-1, 3)

		params = [
			(k, mesh_vertices, mean_tris, mean_verts, mean_tris, transf, transf)
			for k, mesh_vertices in
			enumerate(meshes_vertices)
		]
		displacements_list = Parallel(n_jobs=2)(delayed(displacements)(*p) for p in params)

		all_displacements = np.concatenate(displacements_list)
		disp_min = all_displacements.min()
		disp_max = all_displacements.max()

		print(len(displacements_list))

		params = [
			(rotation_axis, rotation_angle, transf, label, component, k, output_dir, displacement_i,
				mesh_vertices, mean_tris, disp_min, disp_max, cmap)
			for k, (rotation_angle, rotation_axis, mesh_vertices, displacement_i) in enumerate(zip(rotation_angles, rotation_axes, meshes_vertices, displacements_list))
		]
		filenames = Parallel(n_jobs=3)(delayed(generate_frame_open3d_scale)(*p) for p in params)

		path = os.path.join(output_dir, f"vertebra_{label}", "videos")
		os.makedirs(path, exist_ok=True)

		name = f"vertebra{label}_direction{component + 1}_scale.mp4"
		filename = os.path.join(path, name)

		frames_fixed = [fix_frame(f,1080,int(1920/2)) for f in filenames]

		clip = ImageSequenceClip(frames_fixed, fps=fps)
		clip.write_videofile(filename, codec="libx264",
								audio=False,  # no audio stream
								ffmpeg_params=["-pix_fmt", "yuv420p", "-profile:v", "baseline"])

		for img in filenames:
			os.remove(img)




def vis_pca_samples_o3d(Vt, S, mean, norms, label, num_sample):
	# Create images and videos of generating shapes

	cmap = cm.get_cmap("RdYlBu")

	fps = 10

	num_frames = 120

	axes = np.array([[0, 1, 0], [1, 0, 0]])
	rotation_axes = np.repeat(axes, [num_frames / 2, num_frames / 2], axis=0)

	rotation_angles = np.arange(0, 2 * np.pi, 2 * np.pi / (num_frames / 2))
	rotation_angles = np.concatenate((rotation_angles, rotation_angles))

	num_samples = 8
	num_p_components = 50
	p_components = Vt[0:num_p_components,:]
	var = S[0:num_p_components]

	i = 0
	j = 0

	mean_pc = trimesh.points.PointCloud(np.asarray(mean))
	mean_norms = norms.mean(axis=0).reshape(-1, 3)
	mean_pc.vertex_normals = mean_norms.reshape(-1, 3)

	mean_vertices, mean_triangles, kdt = get_pca_meshes(mean_pc, np.asarray([0, 0, 0]), label)

	print("Doing KNN search")

	meshes_vertices = []
	meshes_triangles = []
	transfs = []
	for _ in range(num_samples):
		new_pc = np.empty_like(mean_vertices)
		zs = np.random.randn(num_p_components)
		scales = var / np.sqrt(num_sample - 1)
		pd = np.dot(zs * scales, p_components).reshape(-1,3)
		# coeffs = zs * var/(np.sqrt(len(S))-1)
		# * np.sqrt(eigenvals)
		for n, p in enumerate(mean_vertices):
			cnt, idx, dist = kdt.search_knn_vector_3d(p, 5)
			weights = 1.0 / (np.sqrt(dist) + 1e-8)
			weights /= weights.sum()
			# m = idx[0]
			direction = np.sum(weights[:, None] * pd[idx], axis=0)
			new_pc[n] = p + direction

		# new_points = mean.reshape(-1) + coeffs @ p_components
		# new_points = new_points.reshape(-1,3)
		new_points = new_pc.reshape(-1, 3)
		meshes_vertices.append(new_points)
		meshes_triangles.append(mean_triangles)
		# pc = trimesh.points.PointCloud(new_points)
		# pcs.append(pc)

		# transf = np.asarray([3.5*i, 3.5*j,0])
		transf = np.asarray([90 * j, 90 * i, 0])
		i = i + 1
		if i > 1:
			i = 0
			j = j + 1
		transfs.append(transf)

	# mean_pc = trimesh.points.PointCloud(mean.reshape(-1,3))
	meshes_vertices.append(mean_vertices)
	meshes_triangles.append(mean_triangles)
	transfs.append(np.asarray([0,0,0]))

	# results = Parallel(n_jobs=2)(delayed(get_pca_meshes)(pc, transf, label) for pc, transf in zip(pcs, transfs))

	# meshes_vertices, meshes_triangles = zip(*results)

	mean_verts = meshes_vertices[-1]
	mean_tris = meshes_triangles[-1]

	params = [
		(k, mesh_vertices, mesh_triangles, mean_verts, mean_tris, transf, transfs[-1])
		for k, (mesh_vertices, mesh_triangles, transf) in
		enumerate(zip(meshes_vertices[:-1], meshes_triangles[:-1], transfs[:-1]))
	]
	displacements_list = Parallel(n_jobs=2)(delayed(displacements)(*p) for p in params)

	all_displacements = np.concatenate(displacements_list)
	disp_min = all_displacements.min()
	disp_max = all_displacements.max()

	# displacements_list.append(np.zeros(len(mean_vertices)))

	meshes_vertices = meshes_vertices[:-1]
	meshes_triangles = meshes_triangles[:-1]

	params = [
		(rotation_axis, rotation_angle, transfs, label, k, output_dir, displacements_list, meshes_vertices,
		 meshes_triangles, disp_min, disp_max, cmap)
		for k, (rotation_angle, rotation_axis) in enumerate(zip(rotation_angles, rotation_axes))
	]

	filenames = Parallel(n_jobs=3)(delayed(generate_samples_frame_open3d)(*p) for p in params)

	path = os.path.join(output_dir, f"vertebra_{label}", "videos")
	os.makedirs(path, exist_ok=True)

	name = f"vertebra{label}_samples.mp4"
	filename = os.path.join(path, name)

	frames_fixed = [fix_frame(f,1080,1920) for f in filenames]

	clip = ImageSequenceClip(frames_fixed, fps=fps)
	clip.write_videofile(filename, codec="libx264",
						 audio=False,  # no audio stream
						 ffmpeg_params=["-pix_fmt", "yuv420p", "-profile:v", "baseline"])

	for img in filenames:
		os.remove(img)


def generate_samples_frame_open3d(rotation_axis, rotation_angle, transfs, label, k, output_dir, displacements_list, meshes_vertices,
		 meshes_triangles, disp_min, disp_max, cmap):
	# helper for generating shapes

	meshes = []
	for (vertices, triangles, transf) in zip(meshes_vertices, meshes_triangles, transfs):
		mesh = o3d.geometry.TriangleMesh()
		mesh.vertices = o3d.utility.Vector3dVector(vertices)
		mesh.triangles = o3d.utility.Vector3iVector(triangles)
		mesh.compute_vertex_normals()
		mesh.translate(transf)
		meshes.append(mesh)

	R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_angle * np.array(rotation_axis))

	vis = o3d.visualization.Visualizer()
	vis.create_window(visible=False, width=1920, height=1080)

	for mesh, displacement in zip(meshes, displacements_list):
		disp_norm = displacement / (max(abs(np.min(displacement)), abs(np.max(displacement))) + 1e-8)
		disp_norm = (disp_norm + 1) / 2
		# print(disp_norm)
		colors = cmap(disp_norm)[:, :3]
		mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
		vis.add_geometry(mesh)
		mesh.rotate(R, center=mesh.get_center())
		vis.update_geometry(mesh)

	render_option = vis.get_render_option()
	render_option.light_on = True
	render_option.mesh_show_back_face = True
	render_option.mesh_show_wireframe = True
	render_option.mesh_shade_option = o3d.visualization.MeshShadeOption.Color
	render_option.light_on = False

	ctr = vis.get_view_control()
	# vis.reset_view_point(True)
	for _ in range(20):  # decrease FOV repeatedly
		ctr.change_field_of_view(step=-1.0)
	ctr.set_up([0, 1, 0])
	ctr.set_zoom(.35)

	vis.poll_events()
	vis.update_renderer()

	path = os.path.join(output_dir, f"vertebra_{label}", "images", f"samples")
	os.makedirs(path, exist_ok=True)
	name = f"vertebra{label}_samples_{k}.png"
	filename = os.path.join(path, name)

	vis.capture_screen_image(filename, do_render=True)
	root, _ = os.path.splitext(filename)
	if k % 5 == 0:
		vis.capture_screen_image(root + '.jpg', do_render=True)
	vis.destroy_window()

	return filename


def extract_string(file, string):
	basename = os.path.basename(file)
	match = re.match(string, basename)
	return match.group(1) if match else None

def regression(centered, U, paths, Vt, path, label):
	# preparing csv file for SPIDER regression

	pattern = re.compile(r"^\d+_t[12]_")
	filtered = np.array([f for f in paths if pattern.match(os.path.basename(f))])
	indices = np.where(np.isin(paths, filtered))[0]

	k = 100
	X_points = centered[indices] @ Vt[:k,:].T

	columns_names = []
	for i in range(X_points.shape[1]):
		columns_names.extend([f"PC{i+1}_{label}"])
	X = pd.DataFrame(X_points, columns = columns_names)
	print(X.head())

	id = np.array([extract_string(f, r"^(\d+_t[12])") for f in filtered])

	id_df = pd.DataFrame({"ID": id})
	print(id_df.head())

	overview_path = os.path.join(os.getcwd(), "info", "overview.csv")
	df = pd.read_csv(overview_path, usecols=[0,3,4], header=0)
	df.columns = ['ID', 'Sex','Age']
	df['Sex'] = df['Sex'].str.strip()
	df['Patient'] = df['ID'].str.replace(r'_t[12](_SPACE)?$', '', regex=True).astype("int64")
	print(df.head())

	merged = pd.merge(id_df, df, on="ID", how="left")
	merged['Sex'] = pd.get_dummies(merged[['Sex']], drop_first=True).astype(float)

	merged = pd.concat([merged, X], axis=1)
	print(merged.head())

	merged.to_csv(os.path.join(path, 'merged.csv'), index=False)


def regression_rsna(centered, U, paths, Vt, path, label):
	# preparing csv file for RSNA regression

	k = 500
	X_points = centered @ Vt[:k,:].T

	for x, p in enumerate(paths):
		paths[x] = os.path.splitext(os.path.basename(p))[0]
		paths[x] = paths[x].split("_")[0]

	id_df = pd.DataFrame({"series_id": paths})
	print(id_df.head())

	study_id_path = os.path.join(os.getcwd(), "info", "train_series_descriptions.csv")
	gradings_path = os.path.join(os.getcwd(), "info", "train.csv")

	study_id_df = pd.read_csv(study_id_path)
	study_id_df["series_id"] = study_id_df["series_id"].astype(str)
	print(study_id_df.head())

	columns_names = []
	for i in range(X_points.shape[1]):
		columns_names.extend([f"PC{i+1}_{label}"])
	X = pd.DataFrame(X_points, columns = columns_names)
	print(X.head())

	merged = pd.merge(study_id_df, id_df, on="series_id", how = "right")
	print(merged.head())

	merged = pd.concat([merged, X], axis=1)
	print(merged.head())

	gradings_df = pd.read_csv(gradings_path)

	merged = pd.merge(merged, gradings_df, on="study_id", how="left")
	merged.to_csv(os.path.join(path, 'merged.csv'), index=False)

def regression_rsna_combined(centered, U, paths, Vt, path, label):
	# preparing csv for RSNA regression

    k = 500
    X_points = centered @ Vt[:k,:].T
    
    disc_levels = []
    
    for x, p in enumerate(paths):
        paths[x] = os.path.splitext(os.path.basename(p))[0]
        disc_levels.append(int(paths[x].split("_")[-2].replace("vertebra","")))
        paths[x] = paths[x].split("_")[0]
    
    pca_cols = [f"PC{i+1}" for i in range(k)]
    
    df_pca = pd.DataFrame(X_points, columns=pca_cols)
    df_pca["series_id"] = paths
    df_pca["disc_level"] = disc_levels
    
    print(df_pca.head())
    
    
    id_df = pd.DataFrame({"series_id": paths})
    print(id_df.head())
    
    study_id_path = os.path.join(os.getcwd(), "info", "train_series_descriptions.csv")
    gradings_path = os.path.join(os.getcwd(), "info", "train.csv")
    
    study_id_df = pd.read_csv(study_id_path)
    study_id_df["series_id"] = study_id_df["series_id"].astype(str)
    print(study_id_df.head())
    
    gradings_df = pd.read_csv(gradings_path)
    
    gradings_long = gradings_df.melt(id_vars="study_id", var_name="condition_level", value_name="grading")
    
    split_cols = gradings_long["condition_level"].str.rsplit("_", n=2, expand=True)
    gradings_long["condition"] = split_cols[0]
    gradings_long["disc_level"] = split_cols[1] + "_" + split_cols[2]
    gradings_long = gradings_long[["study_id", "condition", "disc_level", "grading"]]
    
    disc_groups = {
        # "l5_s1": [45, 100],
        "l4_l5": [44, 45, 95],
        "l3_l4": [43, 44, 94],
        "l2_l3": [42, 43, 93],
        "l1_l2": [41, 42, 92]
    }
        
    print(gradings_long.head())
    
    level_groups = {
    # (45, 100): ["upper", "disc"],   
    (44, 45, 95): ["upper", "lower", "disc"],
    (43, 44, 94): ["upper", "lower", "disc"],
    (42, 43, 93): ["upper", "lower", "disc"],
    (41, 42, 92): ["upper", "lower", "disc"],
    }
    
    rows = []
    
    for series_id, df_sub in df_pca.groupby("series_id"):
    
        for disc_name, levels in disc_groups.items():
            row = {
                "series_id": series_id,
                "disc_level": disc_name
            }

            for i, level in enumerate(levels):
                subset = df_sub[df_sub["disc_level"] == level].copy()

                if subset.empty:
                    continue

                if len(levels) == 2:
                    pos = ["upper", "disc"][i]
                else:
                    pos = ["upper", "lower", "disc"][i]
    
                for pc in pca_cols:
                    row[f"{pc}_{pos}"] = subset.iloc[0][pc]
    
            rows.append(row)
    
    df_disc = pd.DataFrame(rows)
    
    print(df_disc.head())
    
    merged = pd.merge(study_id_df, df_disc, on="series_id", how = "right")
    
    merged = pd.merge(merged, gradings_long, on=["study_id", "disc_level"], how="left")
    merged.to_csv(os.path.join(path, 'merged.csv'), index=False)

if __name__ == "__main__":
	labels = [
			41
			,42,43,44,45,
		   91,92,93,94,95,100
					# ,291,292,293,294,295,2100
				]
	# labels = np.arange(1.0, 101.0, 1.0)
	# labels = [41]
	# labels = [
		# "vertebra"
		# , 
		# "disc"
	# ]

	parser = argparse.ArgumentParser(description="PCA")
	parser.add_argument("input", help="Directory containing registered shapes")
	parser.add_argument("--p", type=str, default="pca", help="Directory to save pca outputs")

	args = parser.parse_args()

	output_dir = args.p

	if os.path.isdir(args.input):
		start = time.time()

		angles = []
			
		el = np.linspace(0,360,80)
		az = np.linspace(0,360,80)
			
		pair = list(zip(az,el))
			
		angles.extend(pair)

		for label in labels:
			if isinstance(label, (int, float)):
				path = os.path.join(output_dir, f"vertebra_{label}")
			else:
				path = os.path.join(output_dir, f"all_{label}")

			if os.path.exists(os.path.join(path, "ps.npy")) and os.path.exists(os.path.join(path, "paths.npy")):
				start1 = time.time()

				ps = np.load(os.path.join(path, "ps.npy"))
				paths = np.load(os.path.join(path, "paths.npy"))
				norms = np.load(os.path.join(path, "norms.npy"))
			

			if clean:
				try:
					os.remove(os.path.join(path, "mean.npy"))
				except Exception as e:
					pass
			
			if os.path.exists(os.path.join(path, "U.npy")) and os.path.exists(os.path.join(path, "mean.npy")):
				centered = np.load(os.path.join(path, "centered.npy"))
				U = np.load(os.path.join(path, "U.npy"))
				S = np.load(os.path.join(path, "S.npy"))
				Vt = np.load(os.path.join(path, "Vt.npy"))
				mean = np.load(os.path.join(path, "mean.npy"))
			else:
				ps = ps.reshape(ps.shape[0], -1, 3)
				centered = ps - ps.mean(axis=1, keepdims=True)

				if scale:
					scales = np.load(os.path.join(path, "scales.npz"))["scales"]
					centered *= scales[:, None, None]

				# Mean mesh used for reconstruction
				mean = centered.mean(axis=0)
				centered = centered.reshape(centered.shape[0], -1)

				# PCA
				X = centered.reshape(centered.shape[0], -1)
				X -= mean.reshape(-1)
				U, S, Vt = np.linalg.svd(X, full_matrices = False)
				
				k = min(ps.shape[1], ps.shape[0])
				
				U = U[:, :k]
				S = S[:k]
				Vt = Vt[:k, :]
				np.save(os.path.join(path, "centered.npy"), centered)
				np.save(os.path.join(path, "U.npy"),U)
				np.save(os.path.join(path, "S.npy"),S)
				np.save(os.path.join(path, "Vt.npy"),Vt)
				np.save(os.path.join(path, "mean.npy"),mean)
			
			print(U.shape)
			print(S.shape)
			print(Vt.shape)
			
			
			# elbow(centered, S, output_dir, label, 1)
			# elbow(centered, S[1:], output_dir, label, 2)

			# axis_plot(centered, U, S, Vt, label, output_dir, paths, [0,1,2], 1)
			# axis_plot(centered, U, S, Vt, label, output_dir, paths, [1,2,3], 2)

			# mesh = trimesh.load_mesh(f"/home/nsivaka1/Documents/Research/Spine/reg_pcs/vertebra{label}_targetmatch/{paths[0]}")
			# mean_tris = mesh.faces

			# vis_mean_o3d(ps, mean, norms, label)
	
			# vis_principal_directions_o3d(Vt, S, mean, norms, label)
			# vis_principal_directions_o3d_scale(Vt, S, mean, mean_tris, norms, label)
			
			num_sample = ps.shape[0]
			# vis_pca_samples_o3d(Vt, S, mean, norms, label, num_sample)

			regression(centered, U, paths, Vt, path, label)
			# regression_rsna(centered, U, paths, Vt, path, label)
			# regression_rsna_combined(centered, U, paths, Vt, path, label)
			
			end1 = time.time()
			
			print(f"Completed Vertebra {label}.")
			print_elapsed(start1, end1)

		end = time.time()
		print("Complete")
		print_elapsed(start,end)



