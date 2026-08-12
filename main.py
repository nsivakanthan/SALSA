import subprocess
import time
import os

def print_elapsed(start, end):
	elapsed = end - start

	hours = int(elapsed // 3600)

	minutes = int((elapsed % 3600) // 60)

	seconds = int(elapsed % 60)

	print(f"[Progress] Elapsed time: {hours}h {minutes}m {seconds}s")

def run_mesh(rawdata, mesh):
	# Extract shapes from MRIs; requires totalspineseg
	start = time.time()
	clean_files()

	subprocess.run(
	f"conda run -n spine python -u mesh_test.py {rawdata} --m {mesh} > output_mesh.txt 2>&1"
	, shell = True, check = True)

	end = time.time()
	print('Finished creating meshes')
	print_elapsed(start,end)
	
def run_reg(mesh, reg, mean):
	# Initial cleaning of shapes and rotational alignment
	# Can perform Hungarian/JV registration
	start = time.time()
	
	subprocess.run(
	f"conda run -n spine python -u reg.py {mesh} --r {reg} --m {mean} --o {outliers} > output_reg.txt 2>&1"
	, shell = True, check = True)
	
	end = time.time()
	print('Finished registering')
	print_elapsed(start,end)
	
	
def run_pca(reg, pca):
	# Obtain PCA modes of variation
	# Images and videos of PCA modes, mean vs samples, generating shapes
	# Create csv files for regression using SPIDER and RSNA datasets
	start = time.time()
	
	subprocess.run(
	f"conda run -n spine python -u pca.py {reg} --p {pca} > output_pca.txt 2>&1"
	, shell = True, check = True)
	
	end = time.time()
	print('Finished creating visualiztions of PCA')
	print_elapsed(start,end)


def run_vis_pca(reg, pca):
	# More pca images and videos
	start = time.time()

	subprocess.run(
		f"conda run -n spine python -u vis_pca.py {reg} --p {pca} > output_vis_pca.txt 2>&1"
		, shell=True, check=True)

	end = time.time()
	print('Finished creating visualiztions of PCA')
	print_elapsed(start, end)

def run_regression_R(pca):
	# Obtain regression results for SPIDER dataset
	start = time.time()

	rscript_path = r"C:\Program Files\R\R-4.4.2\bin\Rscript.exe"
	r_script = os.path.join(os.getcwd(), "regression_spider.R")

	subprocess.run([rscript_path, r_script, os.getcwd(), pca])

	end = time.time()
	print("Finished regression in R")
	print_elapsed(start, end)

def run_regression_R_rsna(pca):
	# Obtain regression results for RSNA dataset
	start = time.time()

	rscript_path = r"C:\Program Files\R\R-4.4.2\bin\Rscript.exe"
	r_script = os.path.join(os.getcwd(), "regression_rsna.R")

    # "regression_rsna_full.R"
	subprocess.run([rscript_path, r_script, os.getcwd(), pca])

	end = time.time()
	print("Finished regression in R")
	print_elapsed(start, end)

def run_vis_reg(reg, pca):
	# Visualization of regression directions for SPIDER dataset
	start = time.time()

	subprocess.run(
		f"conda run -n spine python -u vis_reg.py {reg} --p {pca} > output_vis_reg.txt 2>&1"
		, shell=True, check=True)

	end = time.time()
	print('Finished creating visualiztions of PCA Regression')
	print_elapsed(start, end)

def run_vis_reg_rsna(reg, pca):
	# Visualization of regression directions for RSNA dataset
	start = time.time()

	subprocess.run(
		f"conda run -n spine python -u vis_reg_rsna.py {reg} --p {pca} > output_vis_reg.txt 2>&1"
		, shell=True, check=True)

	end = time.time()
	print('Finished creating visualiztions of PCA Regression')
	print_elapsed(start, end)

def clean_files():
	subprocess.run(
	f"del /s *step2*.nii.gz"
	, shell = True, check = True)
	
	subprocess.run(
	f"del /s *step1*.nii.gz"
	, shell = True, check = True)
	
	subprocess.run(
	f"del /s *_reg*.nii.gz"
	, shell = True, check = True)
	
	subprocess.run(
	f"del /s *_sc*.nii.gz"
	, shell = True, check = True)

def main(raw_data, mesh, reg, mean, pca):
	# run_mesh(rawdata, mesh)
    
	# run_reg(mesh, reg, mean)
	
	# run_pca(reg, pca)

	# run_vis_pca(reg, pca)

	# run_regression_R(pca)

	# run_regression_R_rsna(pca)

	# run_vis_reg(reg, pca)

	# run_vis_reg_rsna(reg, pca)

	

rawdata = "data/rawdata/"
mesh = "meshes_rsna"
reg = "reg_pcs"
mean = "mean_pcs"
pca = "pca_rsna"

start = time.time()

main(rawdata, mesh, reg, mean, pca)

end = time.time()

print('Complete')
print_elapsed(start,end)
