import os
import subprocess
import argparse
import glob
import shutil
import time
from joblib import Parallel, delayed
from more_itertools import chunked

os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "6"

import ants
import numpy as np
from skimage import measure
import trimesh
import nibabel as nib
import vtk
from vtk.util import numpy_support
from scipy.ndimage import distance_transform_edt


def print_elapsed(start, end):
    elapsed = end - start
	
    hours = int(elapsed // 3600)
	
    minutes = int((elapsed % 3600) // 60)
	
    seconds = int(elapsed % 60)
	
    print(f"[Progress] Elapsed time: {hours}h {minutes}m {seconds}s")

def segment_spinal_cord(input_nii, k):
    print(f"\n[INFO] Segmenting spine: {input_nii}", flush = True)

    input_dir = os.path.dirname(input_nii)
    input_base = os.path.basename(input_nii).replace(".nii.gz", "").replace(".nii", "")
    seg_name = f"{input_base}_step2_output.nii.gz"
    seg_path = os.path.join(input_dir, seg_name)
    
    if k in [0,1,2,3,4]:
        subprocess.run([
    	    f"CUDA_VISIBLE_DEVICES=1 SCT_USE_GPU=1 sct_deepseg totalspineseg -i {input_nii}"
        ], shell = True, check = True)
    else:
        subprocess.run([
    	    f"CUDA_VISIBLE_DEVICES=0 SCT_USE_GPU=0 sct_deepseg totalspineseg -i {input_nii}"
        ], shell = True, check = True)    

    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"Segmentation output not found: {seg_path}")
    
    print(f"[INFO] Segmentation saved to: {seg_path}")
    return seg_path


def extract_spinal_cord_boundary(label_np, label, label_img, input_nii, output_dir, sub_folder, base_name, threshold = 8):
    disc_labels = [91, 92, 93, 94, 95, 100]

    disc_mask = np.isin(label_np, disc_labels).astype(np.uint8)
    canal_mask = (label_np == label)

    # get nearest-disc indices
    dist, indices = distance_transform_edt(
        ~disc_mask,
        return_indices=True,
        sampling=label_img.spacing
    )

    # nearest disc label for each voxel
    nearest_disc = label_np[
        indices[0],
        indices[1],
        indices[2]
    ]

    dist_top = distance_transform_edt(
        ~(label_np == 91),
        sampling=label_img.spacing
    )
    dist_bottom = distance_transform_edt(
        ~(label_np == 100),
        sampling=label_img.spacing
    )

    for disc_label in disc_labels:

        # Voronoi region intersect canal
        region = (nearest_disc == disc_label) & canal_mask

        # Apply distance cutoff ONLY to top and bottom discs
        if disc_label == 91:
            region = region & (dist_top <= threshold)

        elif label == 100:
            region = region & (dist_bottom <= threshold)

        # Step 1: Signed distance field
        print("[INFO] Computing signed distance field...", flush=True)

        outside = distance_transform_edt(~region)
        inside = distance_transform_edt(region)
        signed_distance = (outside - inside).astype(np.float32)

        # Step 2: Convert numpy → vtkImageData
        print("[INFO] Converting to VTK image...", flush=True)

        img = nib.load(input_nii)
        spacing = img.header.get_zooms()[:3]

        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(signed_distance.shape[::-1])
        vtk_image.SetSpacing(spacing)

        vtk_array = numpy_support.numpy_to_vtk(
            signed_distance.ravel(order="F"),
            deep=True,
            array_type=vtk.VTK_FLOAT
        )

        vtk_image.GetPointData().SetScalars(vtk_array)

        # Step 3: Flying Edges
        print("[INFO] Running Flying Edges...", flush=True)

        flying_edges = vtk.vtkFlyingEdges3D()
        flying_edges.SetInputData(vtk_image)
        flying_edges.SetValue(0, 0.0)
        flying_edges.Update()

        mesh = flying_edges.GetOutput()

        triangle_filter = vtk.vtkTriangleFilter()
        triangle_filter.SetInputData(mesh)
        triangle_filter.Update()

        mesh = triangle_filter.GetOutput()

        # Step 4: Convert vtk → numpy
        print("[INFO] Converting mesh to numpy...", flush=True)

        vtk_points = mesh.GetPoints()
        vtk_faces = mesh.GetPolys()

        verts = numpy_support.vtk_to_numpy(vtk_points.GetData())

        faces = numpy_support.vtk_to_numpy(
            vtk_faces.GetData()
        ).reshape(-1, 4)[:, 1:]

        # --------------------------------------------------
        # Step 5: APPLY AFFINE
        # --------------------------------------------------

        verts = nib.affines.apply_affine(img.affine, verts)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        out_path1 = os.path.join(output_dir, sub_folder, f"{base_name}_vertebra{label}_{disc_label}.ply")

        os.makedirs(os.path.dirname(out_path1), exist_ok=True)

        save_mesh_pointcloud(verts, faces, out_path1)



def extract_vertebra_boundary(label_nii, vertebra_label, label_img, input_nii):

    print(f"[INFO] Extracting vertebra label: {vertebra_label}", flush=True)

    vertebra_mask = (label_nii == vertebra_label)

    if np.sum(vertebra_mask) == 0:
        raise ValueError(f"Label {vertebra_label} not found in image.")

    # Step 1: Signed distance field
    print("[INFO] Computing signed distance field...", flush=True)

    outside = distance_transform_edt(~vertebra_mask)
    inside  = distance_transform_edt(vertebra_mask)
    signed_distance = (outside - inside).astype(np.float32)

    # Step 2: Convert numpy → vtkImageData
    print("[INFO] Converting to VTK image...", flush=True)

    img = nib.load(input_nii)
    spacing = img.header.get_zooms()[:3]

    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(signed_distance.shape[::-1])
    vtk_image.SetSpacing(spacing)

    vtk_array = numpy_support.numpy_to_vtk(
        signed_distance.ravel(order="F"),
        deep=True,
        array_type=vtk.VTK_FLOAT
    )

    vtk_image.GetPointData().SetScalars(vtk_array)

    # Step 3: Flying Edges
    print("[INFO] Running Flying Edges...", flush=True)

    flying_edges = vtk.vtkFlyingEdges3D()
    flying_edges.SetInputData(vtk_image)
    flying_edges.SetValue(0, 0.0)
    flying_edges.Update()

    mesh = flying_edges.GetOutput()

    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(mesh)
    triangle_filter.Update()

    mesh = triangle_filter.GetOutput()

    # Step 4: Convert vtk → numpy
    print("[INFO] Converting mesh to numpy...", flush=True)

    vtk_points = mesh.GetPoints()
    vtk_faces = mesh.GetPolys()

    verts = numpy_support.vtk_to_numpy(vtk_points.GetData())

    faces = numpy_support.vtk_to_numpy(
        vtk_faces.GetData()
    ).reshape(-1, 4)[:, 1:]

    # --------------------------------------------------
    # Step 5: APPLY AFFINE HERE
    # --------------------------------------------------

    verts_physical = nib.affines.apply_affine(img.affine, verts)

    return verts_physical, faces
    

def save_mesh_pointcloud(verts, faces, out_path1):    
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(out_path1)
    
    print(f"[INFO] Mesh saved to: {out_path1}")
    


def clean_files(input_nii):
	directory = os.path.dirname(input_nii)
	
	subprocess.run([
	f"find {directory} -name *step2*.nii.gz -delete"
	], shell = True, check = True)
	
	subprocess.run([
	f"find {directory} -name *step1*.nii.gz -delete"
	], shell = True, check = True)
	
	subprocess.run([
	f"find {directory} -name *reg*.nii.gz -delete"
	], shell = True, check = True)
	
	subprocess.run([
	f"find {directory} -name *sc*.nii.gz -delete"
	], shell = True, check = True)

def check_size(source_nii_path):
    try:
    	start = time.time()
    	
    	source_nii = ants.image_read(source_nii_path)
    	
    	spacing = source_nii.spacing
    	
    	if any(dim > 800 for dim in source_nii.shape) or all(dim > 500 for dim in source_nii.shape):
    	    print(f"File size too large: {source_nii_path}")
    	    return None
    
    	end = time.time()
    
    	print_elapsed(start, end)
    	
    	return source_nii_path

    except RuntimeError as e:
    	print(f"Runtime Error on file {source_nii_path}: {e}")
    	return None
    except Exception as e:
        print(f"[Error] Failed on {source_nii_path}: {e}")
        return None
        
    
def clean_tmp(tmp_files):
    for f in tmp_files:
        subprocess.run([
        f"rm -f {f}"
        ], shell = True, check = True)


def process_single_file(input_nii, output_dir, k):
    try:
        base_name = os.path.splitext(os.path.basename(input_nii))[0].replace(".nii.gz","").replace(".nii","").replace("_reg","")
        sub_folder = os.path.basename(os.path.dirname(input_nii))

        seg_labeled_nii = segment_spinal_cord(input_nii, k)
        
        label_img = ants.image_read(seg_labeled_nii)

        label_np = label_img.numpy()

        vertebra_labels = np.unique(label_np)
        
        if len(vertebra_labels) < 2:
        	raise ValueError(f"Labels not found.")

        vertebra_labels = vertebra_labels[(vertebra_labels>=1) & (vertebra_labels <= 100)]

        for label in vertebra_labels:
            try:
                if label == 2:
                    extract_spinal_cord_boundary(label_np, label, label_img, input_nii, output_dir, sub_folder, base_name, threshold = 8)

                else:
                    verts, faces = extract_vertebra_boundary(label_np, label, label_img, input_nii)

                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)

                    out_path1 = os.path.join(output_dir, sub_folder, f"{base_name}_vertebra{label}.ply")

                    os.makedirs(os.path.dirname(out_path1), exist_ok=True)

                    save_mesh_pointcloud(verts, faces, out_path1)
        
        
            except ValueError as e:
                print(f"[WARNING] {e}")
        clean_files(input_nii)

    except Exception as e:
        print(f"[Error] Failed on {input_nii}: {e}")

def batch_process_folder(folder_path, output_dir):
    nii_files = sorted(glob.glob(os.path.join(folder_path, "**/*.nii.gz"), recursive=True))
    if not nii_files:
        print(f"[WARNING] No .nii.gz files found in {folder_path}")
        return
    
    new_nii_files = Parallel(n_jobs=20)(delayed(check_size)(nii_file) for nii_file in nii_files)
    
    new_nii_files = [r for r in new_nii_files if r is not None]
    
    params = [(nii_file, output_dir, k) for k, nii_file in enumerate(nii_files)]
    
    Parallel(n_jobs=12)(delayed(process_single_file)(*p) for p in params)
    
        
if __name__ == "__main__":
	
    parser = argparse.ArgumentParser(description="Spine segmentation + vertebra mesh extraction using SCT")
    parser.add_argument("input", help="Input .nii.gz file or directory containing NIfTI files")
    parser.add_argument("--m", type=str, default="meshes", help="Directory to save mesh outputs")

    args = parser.parse_args()

    if os.path.isfile(args.input):
        process_single_file(args.input, args.m)
    elif os.path.isdir(args.input):
        batch_process_folder(args.input, args.m)
    else:
        print(f"[ERROR] Invalid input path: {args.input}")
