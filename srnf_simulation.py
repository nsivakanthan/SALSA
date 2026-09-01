import numpy as np
import open3d as o3d
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse import coo_matrix
import sys
import os
import random
sys.path.append('../H2_SurfaceMatch')
import utils.input_output as input_output
import SRNF_match as matching
# from enr.rigid_match import*
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from joblib import Parallel, delayed
import pymeshlab as pml
import pyvista as pv
import pyacvd

def uniform_resample(mesh):
    V, F = mesh

    ms = pml.MeshSet()
    mesh = pml.Mesh(vertex_matrix=V, face_matrix=F)
    ms.add_mesh(mesh, "input_mesh")

    # Apply uniform mesh resampling
    ms.generate_resampled_uniform_mesh()
    ms.meshing_remove_connected_component_by_face_number()
    result = ms.current_mesh()

    V_resampled = result.vertex_matrix()
    F_resampled = result.face_matrix()

    return V_resampled, F_resampled

def acvd_resample(mesh):
    V, F = mesh
    faces_pv = np.hstack([np.full((F.shape[0], 1), 3), F]).flatten()
    pv_mesh = pv.PolyData(V, faces_pv)
    # pv_mesh.plot(color='w', show_edges=True)
    clus = pyacvd.Clustering(pv_mesh)
    # clus.subdivide(3)
    clus.cluster(1000)
    # plot clustered
    # clus.plot()

    remesh = clus.create_mesh()
    remesh.clean(inplace=True)
    new_V = remesh.points
    new_F = remesh.faces.reshape((-1, 4))[:, 1:]

    return new_V, new_F
    
def compute_cotangent_laplacian(vertices, faces):
    """
    Compute the cotangent Laplacian matrix for a triangular mesh.

    Parameters
    ----------
    vertices : (n, 3) array_like
        Array of vertex coordinates.
    faces : (m, 3) array_like
        Array of triangle indices (each row is a triangle).

    Returns
    -------
    L : scipy.sparse.coo_matrix
        Sparse symmetric cotangent Laplacian matrix.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must be of shape (n, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be of shape (m, 3)")

    # Extract vertex positions for each triangle
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Compute edge vectors
    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2

    # Compute triangle areas using cross product
    face_areas = np.linalg.norm(np.cross(e0, -e2), axis=1) * 0.5
    if np.any(face_areas == 0):
        raise ValueError("Degenerate triangle with zero area detected.")

    # Compute cotangents of angles
    cot0 = -np.einsum('ij,ij->i', e2, e0) / np.linalg.norm(np.cross(e2, e0), axis=1)
    cot1 = -np.einsum('ij,ij->i', e0, e1) / np.linalg.norm(np.cross(e0, e1), axis=1)
    cot2 = -np.einsum('ij,ij->i', e1, e2) / np.linalg.norm(np.cross(e1, e2), axis=1)

    # Build COO sparse matrix entries
    I = np.hstack([faces[:, 1], faces[:, 2], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1]])
    J = np.hstack([faces[:, 2], faces[:, 1], faces[:, 0], faces[:, 2], faces[:, 1], faces[:, 0]])
    W = 0.5 * np.hstack([cot0, cot0, cot1, cot1, cot2, cot2])

    # Assemble symmetric weight matrix
    W_sparse = coo_matrix((W, (I, J)), shape=(len(vertices), len(vertices)))

    # Degree matrix (sum of weights per vertex)
    diag_w = np.array(W_sparse.sum(axis=1)).flatten()
    L = coo_matrix((np.hstack([W, -diag_w]),
                    (np.hstack([I, np.arange(len(vertices))]),
                     np.hstack([J, np.arange(len(vertices))]))),
                   shape=(len(vertices), len(vertices)))

    return L

def shuffle(V, F):
    N = V.shape[0]

    # Shuffle vertices
    perm_v = np.random.permutation(N)
    V_shuffled = V[perm_v]

    # Build inverse mapping: old index -> new index
    inv_perm_v = np.zeros(N, dtype=int)
    inv_perm_v[perm_v] = np.arange(N)

    # Update faces to match new vertex indices
    F_updated = inv_perm_v[F]

    return V_shuffled, F_updated, perm_v

def unshuffle(V_shuffled, F_updated, perm_v):
    inv_perm = np.argsort(perm_v)

    V_restored = V_shuffled[inv_perm]
    F_restored = perm_v[F_updated]

    return V_restored, F_restored


def run_simulation(file, source_folder, label = 41):
    [VT,FT,FunT] = input_output.loadData(os.path.join(source_folder,file))

    target = [VT,FT]

    VT, FT = uniform_resample_mesh(target)
    
    L = compute_cotangent_laplacian(VT, FT)
    k = 50  # number of modes
    
    eigvals, eigvecs = spla.eigsh(L, k=k, sigma=0)
    
    scale = np.linalg.norm(VT.max(axis=0) - VT.min(axis=0))
    coeffs = np.random.normal(scale=.15 * scale, size=k)
    
    deformation = eigvecs @ coeffs   # (V,)
    deformation = deformation[:, None]  # apply to xyz
    
    VS = VT + deformation
    
    errors = np.linalg.norm(VT - VS, axis = 1)
    mean_error = errors.mean()
    
    VS, FS, perm_vs = shuffle(VS, FT)
    
    target = [VT, FT]
    source = [VS, FS]
    
    if label in [41,42,43,44]:
        sig_geom_val1 = .2
        sig_geom_val2 = .1
        tri_num = int(FT.shape[0]*15/16)
    elif label in [45]:
        sig_geom_val1 = .1
        sig_geom_val2 = .5
        tri_num = int(FT.shape[0]*4/16)
    elif label in [91,92,93,94,95]:
        sig_geom_val1 = .1
        sig_geom_val2 = .1
        tri_num = int(FT.shape[0]*4/16)
    elif label in [100]:
        sig_geom_val1 = .2
        sig_geom_val2 = .2
        tri_num = int(FT.shape[0]*4/16)
        
    [V0,F0]= input_output.decimate_mesh(VT,FT,tri_num)
    source_init = [V0,F0]
    target_init = [V0,F0]
    
    weight_MCV=0 #no SRCF term = 0
        
    parameters1 = {'weight_MCV':weight_MCV,'weight_coef_dist_S': 10**3, 'weight_coef_dist_T': 10**3,\
              'kernel_geom':'gaussian','sig_geom': sig_geom_val1,'kernel_grass':'binet',\
              'sig_grass':1 ,'max_iter': 50,'use_fundata':0}
    
    paramlist=[parameters1]
    
    num_params = len(paramlist)
    
    # Run multiresolution matching 
    f0,f1,Tri,Funct,En,Dic = matching.MultiResMatching(source,target,source_init,target_init,paramlist)
    
    match_source=[f0[num_params-1],Tri[num_params-1]]
    match_target=[f1[num_params-1],Tri[num_params-1]]
    
    VS_orig, FS_orig = unshuffle(VS, FS, perm_vs)
    
    tree = cKDTree(match_target[0])
    dist, idx = tree.query(VT)
    idx_base = idx.copy()
    VT_match = match_target[0][idx]
    print(idx)
    tree = cKDTree(match_source[0])
    dist, idx = tree.query(VS_orig)
    VS_match = match_source[0][idx]
    
    tree = cKDTree(VT_match)
    dist, idx_target = tree.query(VT)
    
    tree = cKDTree(VS_match)
    dist, idx_source = tree.query(VS_orig)
    
    mismatches = idx_target != idx_source
    
    diff_srnf = np.sum(mismatches)
    
    pct_error_srnf = diff_srnf/source[0].shape[0]
    
    errors = np.linalg.norm(VT[idx_target] - VS_orig[idx_source], axis = 1)
    mean_error_srnf = errors.mean()
    
    C = cdist(VT,VS)
    row_ind, col_ind = linear_sum_assignment(C)
    
    tree = cKDTree(VS[col_ind])
    dist, idx = tree.query(VS_orig)
    
    mismatches = idx != idx_base
    
    diff_jv = np.sum(mismatches)
    
    pct_error_jv = diff_jv/source[0].shape[0]
    
    errors = np.linalg.norm(VT[row_ind] - VS[col_ind], axis=1)
    mean_error_jv = errors.mean()

    return pct_error_srnf, pct_error_jv, mean_error_srnf, mean_error_jv, mean_error


label = 41
source_folder = "reg_pcs/vertebra41"

files= sorted(os.listdir(source_folder))

subset = random.sample(files, 100)
print(subset)

results = Parallel(n_jobs=1)(delayed(run_simulation)(file, source_folder) for file in subset)

srnf_pct_errors = []
jv_pct_errors = []
srnf_mean_errors = []
jv_mean_errors = []
true_mean_errors = []

for r in results:
    srnf_pct_errors.append(r[0])
    jv_pct_errors.append(r[1])
    srnf_mean_errors.append(r[2])
    jv_mean_errors.append(r[3])
    true_mean_errors.append(r[4])
    del r

srnf_pct_errors = np.array(srnf_pct_errors)
jv_pct_errors = np.array(jv_pct_errors)
srnf_mean_errors = np.array(srnf_mean_errors)
jv_mean_errors = np.array(jv_mean_errors)
true_mean_errors = np.array(true_mean_errors)

def summarize(x):
    x = np.array(x)
    mean = x.mean()
    std = x.std(ddof=1)
    se = std / np.sqrt(len(x))
    ci_lower = mean - 1.96 * se
    ci_upper = mean + 1.96 * se
    return mean, std, se, ci_lower, ci_upper

# summaries
srnf_pct_stats = summarize(srnf_pct_errors)
jv_pct_stats = summarize(jv_pct_errors)

diff_srnf = true_mean_errors - srnf_mean_errors
diff_jv = true_mean_errors - jv_mean_errors

diff_srnf_stats = summarize(diff_srnf)
diff_jv_stats = summarize(diff_jv)

def format_stats(name, stats):
    mean, std, se, ci_l, ci_u = stats
    return (
        f"{name}:\n"
        f"  Mean = {mean:.6f}\n"
        f"  Std  = {std:.6f}\n"
        f"  SE   = {se:.6f}\n"
        f"  95% CI = [{ci_l:.6f}, {ci_u:.6f}]\n"
    )

summary = (
    format_stats("SRNF % Error", srnf_pct_stats) + "\n" +
    format_stats("JV % Error", jv_pct_stats) + "\n" +
    format_stats("True - SRNF Mean Error", diff_srnf_stats) + "\n" +
    format_stats("True - JV Mean Error", diff_jv_stats)
)

print(summary)

with open("srnf_error_results.txt", "w") as f:
    f.write(summary)


