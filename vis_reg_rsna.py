import os
import argparse
import glob
import time
from joblib import Parallel, delayed
import re
import math
import sys, traceback
import io

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
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from PIL import Image
import cv2


def print_elapsed(start, end):
	elapsed = end - start
	hours = int(elapsed // 3600)
	minutes = int((elapsed % 3600) // 60)
	seconds = int(elapsed % 60)
	print(f"[Progress] Elapsed time: {hours}h {minutes}m {seconds}s")


def get_filenames(label, file_path="reg_pointclouds"):
	folder_path = os.path.join(os.getcwd(), file_path)
	suffix = f"**/*vertebra{label}*.ply"
	ply_files = sorted(glob.glob(os.path.join(folder_path, suffix), recursive=True))

	return ply_files

def fix_frame(filename):
	frame = np.array(Image.open(filename))
	h, w, _ = frame.shape

	target_height, target_width = 1080, 1920

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

def vertebra_cmap():
	"""
	Dark diverging colormap with slightly lighter vertebra-brown midpoint and 10 stops.
	"""
	# Anchor colors
	high_color = np.array([0.1, 0.2, 0.5])  # dark blue
	mid_color = np.array([0.55, 0.38, 0.28])  # slightly lighter vertebra brown
	low_color = np.array([0.8, 0.1, 0.1])  # rusty red

	# Generate 10 stops
	n_stops = 10
	colors = []
	for i in range(n_stops):
		t = i / (n_stops - 1)
		if t < 0.5:
			# interpolate from low_color → mid_color
			t2 = t / 0.5
			c = (1 - t2) * low_color + t2 * mid_color
		else:
			# interpolate from mid_color → high_color
			t2 = (t - 0.5) / 0.5
			c = (1 - t2) * mid_color + t2 * high_color
		colors.append(c)

	colors = np.array(colors)
	# Add alpha = 1
	colors = np.hstack([colors, np.ones((n_stops, 1))])

	return mcolors.LinearSegmentedColormap.from_list("vertebra_diverging_lightbrown", colors)

def get_pca_meshes(pc, transf, label):
    # pc.apply_translation(transf)
    
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(vertex_matrix=np.asarray(pc.vertices), v_normals_matrix=np.asarray(pc.vertex_normals))
    ms.add_mesh(mesh)

    print(label)
    
    if int(label) in [41,42,43,44,45]:
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
    # normals = m.vertex_normal_matrix()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    # mesh.triangle_normals = o3d.utility.Vector3dVector(normals)
    mesh.compute_vertex_normals()
    mesh.translate(transf)
    
    ref_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc.vertices))
    kdt = o3d.geometry.KDTreeFlann(ref_pc)
    
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles), kdt

def displacements(k, mesh_vertices, mesh_triangles, mean_mesh_vertices, mean_mesh_triangles, transf, transf_mean):
	mesh_ref = trimesh.Trimesh(vertices=np.asarray(mean_mesh_vertices), faces=np.asarray(mean_mesh_triangles))
	mesh_query = trimesh.Trimesh(vertices=np.asarray(mesh_vertices), faces=np.asarray(mesh_triangles))

	# mesh_ref.apply_translation(-transf_mean)
	# mesh_query.apply_translation(-transf)

	# mesh_ref.fix_normals()
	# mesh_query.fix_normals()
	displacement = trimesh.proximity.signed_distance(mesh_ref, mesh_query.vertices)

	return displacement

def generate_frame_open3d(rotation_axis, rotation_angle, transf, condition, k, output_dir, displacement, mesh_vertices, mesh_triangles, disp_min, disp_max, cmap, unique_label, mri_type):

	mesh = o3d.geometry.TriangleMesh()
	mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
	mesh.triangles = o3d.utility.Vector3iVector(mesh_triangles)
	mesh.compute_vertex_normals()
	mesh.translate(transf)


	R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_angle * np.array(rotation_axis))

	vis = o3d.visualization.Visualizer()
	vis.create_window(visible = False, width=1920, height=1080)
	# vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])

	# disp_norm = (displacement - disp_min) / (disp_max - disp_min + 1e-8)
	# disp_norm = displacement / (max(abs(disp_min), abs(disp_max)) + 1e-8)
	disp_norm = displacement / (max(abs(np.min(displacement)), abs(np.max(displacement))) + 1e-8)
	disp_norm = (disp_norm +1)/2
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
	# vis.reset_view_point(True)
	ctr.set_up([1,0,0])
	ctr.set_zoom(.7)
	vis.poll_events()
	vis.update_renderer()

	path = os.path.join(output_dir, f"reg", "images", f"{condition}_{label}_{unique_label}_{mri_type}")
	os.makedirs(path, exist_ok=True)
	name = f"{condition}_{label}_{unique_label}_{mri_type}_{k}.png"
	filename = os.path.join(path, name)

	vis.capture_screen_image(filename, do_render = True)
	root, _ = os.path.splitext(filename)
	if k % 10 == 0:
		vis.capture_screen_image(root + '.jpg', do_render = True)
	vis.destroy_window()

	return filename



def vis_reg_coefs(mean, mean_norms, Vt, S, label, scaling_coefs, condition, unique_label,mri_type):
	# Create images and videos for regression directions of various binary classification problems scaled from the mean horizontally

	components = 10
	num_frames = 120

	axes = np.array([[0,1,0], [1,0,0]])
	rotation_axes = np.repeat(axes, [num_frames/2,num_frames/2], axis = 0)

	rotation_angles = np.arange(0, 2 * np.pi, 2 * np.pi / (num_frames/2))
	rotation_angles = np.concatenate((rotation_angles, rotation_angles))

	fps = 10

	cmap = cm.get_cmap("RdYlBu")

	transf = np.asarray([0, 0, 0])

	mean_pc = trimesh.points.PointCloud(np.asarray(mean).reshape(-1,3))
	mean_pc.vertex_normals = mean_norms.reshape(-1, 3)
	mean_vert, mean_tri, kdt = get_pca_meshes(mean_pc, transf, label)


	pd = np.dot(scaling_coefs, Vt[:components, :]).reshape(-1, 3)

	new_pc = np.empty_like(mean_vert)

	for n, p in enumerate(mean_vert):
		cnt, idx, dist = kdt.search_knn_vector_3d(p, 5)

		weights = 1.0 / (np.sqrt(dist) + 1e-8)
		weights /= weights.sum()
		# m = idx[0]
		direction = np.sum(weights[:,None] * pd[idx], axis=0)

		new_pc[n] = p + direction

	all_displacements = displacements(0, new_pc, mean_tri, mean_vert, mean_tri, transf, transf)

	disp_min = all_displacements.min()
	disp_max = all_displacements.max()

	params = [
		(rotation_axis, rotation_angle, transf, condition, k, output_dir, all_displacements, new_pc,
		 mean_tri, disp_min, disp_max, cmap, unique_label,mri_type)
		for k, (rotation_angle, rotation_axis) in enumerate(zip(rotation_angles, rotation_axes))
	]
	filenames = Parallel(n_jobs=5)(delayed(generate_frame_open3d)(*p) for p in params)

	path = os.path.join(output_dir, f"reg", "videos")
	os.makedirs(path, exist_ok=True)

	name = f"{condition}_{label}_{unique_label}_{mri_type}.mp4"
	filename = os.path.join(path, name)

	clip = ImageSequenceClip(filenames, fps=fps)
	clip.write_videofile(filename, codec="libx264")

	for img in filenames:
		os.remove(img)

def generate_frame_open3d_scale(rotation_axis, rotation_angle, transf, condition, k, output_dir, displacement, mesh_vertices, mesh_triangles, disp_min, disp_max, cmap):

	mesh = o3d.geometry.TriangleMesh()
	mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
	mesh.triangles = o3d.utility.Vector3iVector(mesh_triangles)
	mesh.compute_vertex_normals()
	mesh.translate(transf)


	R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_angle * np.array(rotation_axis))

	# disp_norm = (displacement - disp_min) / (disp_max - disp_min + 1e-8)
	disp_norm = displacement / (max(abs(disp_min), abs(disp_max)) + 1e-8)
	# disp_norm = displacement / (max(abs(np.min(displacement)), abs(np.max(displacement))) + 1e-8)
	disp_norm = (disp_norm +1)/2
	disp_norm = np.asarray(disp_norm)

	orig_colors = cmap(disp_norm)[:, :3]

	# neutral_mask = np.isclose(disp_norm, 0.5, atol=0.01)  # find "neutral" values
	# vertebra_rgb = np.array([0.9, 0.8, 0.6])
	# orig_colors[neutral_mask] = vertebra_rgb
	colors = orig_colors

	mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

	vis = o3d.visualization.Visualizer()
	vis.create_window(visible = False, width=1920, height=1080)
	# vis.get_render_option().background_color = np.array([0.9, 0.9, 0.9])

	vis.add_geometry(mesh)
	mesh.rotate(R, center=mesh.get_center())
	vis.update_geometry(mesh)

	opt = vis.get_render_option()
	opt.mesh_show_back_face = True
	opt.mesh_show_wireframe = True
	opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Color
	opt.light_on = False

	ctr = vis.get_view_control()
	# vis.reset_view_point(True)
	for _ in range(20):  # decrease FOV repeatedly
		ctr.change_field_of_view(step=-1.0)
	ctr.set_up([-1,0,0])
	ctr.set_zoom(.7)

	vis.update_geometry(mesh)
	vis.poll_events()
	vis.update_renderer()

	path = os.path.join(output_dir, f"reg_scale", "images", f"{condition}_{label}")
	os.makedirs(path, exist_ok=True)
	name = f"{condition}_{label}_{k}.png"
	filename = os.path.join(path, name)

	vis.capture_screen_image(filename, do_render = True)
	root, _ = os.path.splitext(filename)
	if k % 2 == 0:
		vis.capture_screen_image(root + '.jpg', do_render = True)
	vis.destroy_window()

	return filename





def vis_reg_coefs_scale(mean, mean_norms, Vt, S, label, scaling_coefs, condition):
	# Create images and videos for regression directions of various binary classification problems scaled from the mean

	components = 50
	num_frames = 120

	axes = np.array([[0,1,0], [1,0,0]])
	rotation_axes = np.repeat(axes, [num_frames/2,num_frames/2], axis = 0)

	rotation_angles = np.arange(0, 2 * np.pi, 2 * np.pi / (num_frames/2))
	rotation_angles = np.concatenate((rotation_angles, rotation_angles))
    
	# rotation_axis = np.array([0,1,1])
	# rotation_angle = np.array([.25 * np.pi])
	# rotation_axis = np.array([0, 1, 0])
	# rotation_angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)
	# rotation_axes = np.repeat(rotation_axis[np.newaxis, :], num_frames, axis=0)

	fps = 10

	cmap = cm.get_cmap("RdYlBu")
	# cmap = darken_cmap("managua", factor = .99)
	# cmap = vertebra_cmap()
	transf = np.asarray([0, 0, 0])

	mean_pc = trimesh.points.PointCloud(np.asarray(mean).reshape(-1,3))
	mean_pc.vertex_normals = mean_norms.reshape(-1, 3)
	mean_vert, mean_tri, kdt = get_pca_meshes(mean_pc, transf, label)

	scalings = [np.linspace(0,s, num_frames) for s in scaling_coefs]

	pcs = []

	for scaling in np.asarray(scalings).T:

		pd = np.dot(scaling, Vt[:components, :]).reshape(-1, 3)

		new_pc = np.empty_like(mean_vert)

		for n, p in enumerate(mean_vert):
			cnt, idx, dist = kdt.search_knn_vector_3d(p, 5)


			weights = 1.0 / (np.sqrt(dist) + 1e-8)
			weights /= weights.sum()

			direction = np.sum(weights[:,None] * pd[idx], axis=0)

			new_pc[n] = p + direction

		pcs.append(new_pc)


	params = [
		(k, pc, mean_tri, mean_vert, mean_tri, transf, transf)
		for k, pc in enumerate(pcs)
	]
	displacements_list = Parallel(n_jobs=2)(delayed(displacements)(*p) for p in params)

	all_displacements = np.stack(displacements_list)

	disp_min = all_displacements.min()
	disp_max = all_displacements.max()

	params = [
		(rotation_axis, rotation_angle, transf, condition, k, output_dir, displacement_i, new_pc,
		 mean_tri, disp_min, disp_max, cmap)
		for k, (rotation_axis, rotation_angle, new_pc, displacement_i) in enumerate(zip(rotation_axes, rotation_angles, pcs, displacements_list))
	]
	filenames = Parallel(n_jobs=5)(delayed(generate_frame_open3d_scale)(*p) for p in params)

	frames_fixed = [fix_frame(f) for f in filenames]

	path = os.path.join(output_dir, f"reg_scale", "videos")
	os.makedirs(path, exist_ok=True)

	name = f"{condition}_{label}.mp4"
	filename = os.path.join(path, name)

	# name2 = f"{condition}_{label}_{unique_label}_{mri_type}.gif"
	# filename2 = os.path.join(path, name2)

	clip = ImageSequenceClip(frames_fixed, fps=fps)
	clip.write_videofile(filename, codec="libx264",
    audio=False,                  # no audio stream
    ffmpeg_params=["-pix_fmt", "yuv420p", "-profile:v", "baseline"])
	# clip.write_gif(filename2, fps=fps)

	for img in filenames:
		os.remove(img)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCA visualizations between vertebrae/discs")
    parser.add_argument("input", help="Directory containing registered point clouds")
    parser.add_argument("--p", type=str, default="pca", help="Directory to save pca outputs")
    
    args = parser.parse_args()
    
    output_dir = args.p
    path = os.path.join(output_dir, f"reg")
    os.makedirs(path, exist_ok=True)
    
    csv_paths = sorted(glob.glob(os.path.join(output_dir,"models_run", "**/*.csv"), recursive=True))
    
    if os.path.isdir(args.input):
        for csv_path in csv_paths:
            coefs = pd.read_csv(csv_path)
            labels = coefs.iloc[:,0].str.split('_').str[-1].unique()
            labels = [v for v in labels if re.fullmatch(r'\d+(\.\d+)?', v)]
            condition = coefs.columns[1]
            print(csv_path, flush = True)
            print(labels, flush = True)
            print(condition, flush = True)
    
            for label in labels:
                scaling_coefs = coefs.loc[coefs.iloc[:, 0].astype(str).str.contains(f"_{label}")].iloc[:, 1].to_numpy()
                print(scaling_coefs, flush = True)
                suffix = f"mean.npy"
                mean_path = sorted(glob.glob(os.path.join(output_dir, f"vertebra_{label}", suffix), recursive=True))
                suffix = f"Vt.npy"
                Vt_path = sorted(glob.glob(os.path.join(output_dir, f"vertebra_{label}", suffix), recursive=True))
                suffix = f"S.npy"
                S_path = sorted(glob.glob(os.path.join(output_dir, f"vertebra_{label}", suffix), recursive=True))
                suffix = f"norms.npy"
                norms_path = sorted(glob.glob(os.path.join(output_dir, f"vertebra_{label}", suffix), recursive=True))
    
                mean = np.load(mean_path[0])
                Vt = np.load(Vt_path[0])
                S = np.load(S_path[0])
                # scaling_coefs = scaling_coefs * np.sqrt(S[:len(scaling_coefs)])
                scaling_coefs = scaling_coefs * S[:len(scaling_coefs)]/np.sqrt(len(S)-1)
                # print(S[:len(scaling_coefs)])
                norms = np.load(norms_path[0])
                mean_norms = norms.mean(axis=0).reshape(-1, 3)
    
                vis_reg_coefs_scale(mean, mean_norms, Vt, S, label, scaling_coefs, condition)
    
