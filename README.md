# SALSA
Shape Analysis for Lumbar Spine Assessments

This repository contains the code and displays some results of the SALSA pipeline. We apply this pipeline to two publicly available datasets: 1.) RSNA-LumbarDisc dataset 2.) SPIDER dataset.

The main code file is main.py which runs the following files (order matters):

  - 1.) mesh.py - Extracts shapes (triangluar meshes; .ply files) from MRIs (.nii.gz files) using totalspineseg from the spinal cord toolbox (SCT) (must download and use specific version of totalspineseg).
  
  - 2.) reg.py - Cleans and initial alignment of shapes of the lumbar spine. Can also be used to perform Hungarian/Jonker-Volgenant registration of shapes.
  
  - (SRNF registration) After cleaning and initial alignment, perform SRNF registration. More requirements and code described later. 
  
  - 3.) pca.py - Obtain prinicpal modes of variation, visualize these modes, prepare csv files for regression on both SPIDER and RSNA datasets.
  
  - 4.) regression_rsna.R and/or regression_spider.R - Obtain binary and multi-label classification results for various spinal conditions.
  
  - 5.) vis_reg.py, vis_reg_rsna.py - Visualize regression directions to interpret regression results.

Elastic Registration w/ SRNF:
The main theoretical contribution of this work comes from improvement of results using Elastic registration. We employ code developed by ____ and _____ to register vertebra and discs using the SRNF. This required downloading code from their github repository and making minimal adjustments to change which functions were being used. In addition, a specific conda environment must be managed to run their code revolved around the PyKeops package. Their code also requires a linux system to run (can also use WSL). 

The following code files are used for Elastic Registration:
  
  - 1.) srnf_reg.py - Register shapes and obtain means on a per shape level.
  
  - 2.) srnf_reg_all.py - Register shapes and obtain means pooled between vertebrae and discs, respectively.
