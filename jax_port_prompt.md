# Task: Create JAX-compatible DDM_P2P and DDM_M2P

Create `DDM_P2P_JAX` and `DDM_M2P_JAX` — pure JAX reimplementations of `DDM_P2P` and `DDM_M2P` from `DDM/__init__.py`, so they can be used inside `jax.jit` and `jax.vmap`.

## Context

The existing `DDM_P2P` and `DDM_M2P` are PyTorch `nn.Module` classes. I need JAX equivalents so I can use them as optimization objectives inside a JIT-compiled JAX inference loop (Bayes3D / GenJAX). Currently I have to break out of JAX, convert to torch tensors, call DDM, and convert back — this prevents JIT compilation and vmapping over candidate poses.

## What to implement

### `DDM_P2P_JAX(src, tgt, K=5, up_ratio=10, std=0.05, weighted_query=True, beta=3)`
- `src`: (B, N, 3) jax array — source point cloud
- `tgt`: (B, M, 3) jax array — target point cloud
- Returns: scalar loss

Port the logic from `DDM_P2P.forward()` in `DDM/__init__.py`. Key operations:
1. Generate noisy query points around `tgt` (random offsets with given `std`), concatenate with `src`
2. `cal_udf_weights`: for each query point, find K nearest neighbors in both `src` and `tgt`, compute inverse-distance-weighted UDF and UDF gradient
3. Compute `udf_error = |udf_tgt - udf_src|` and `udf_grad_error = sum(|grad_src - grad_tgt|, axis=-1)`
4. If `weighted_query`: weight errors by `exp(-udf_error * beta) * exp(-udf_grad_error * beta)` (detached/stop-gradient weights)
5. Return weighted mean of `(udf_error + udf_grad_error)`

The KNN in PyTorch uses `pytorch3d.ops.knn_points`. In JAX, implement KNN via squared distance matrix + `jax.lax.top_k` (negate distances for top_k to get smallest K).

### `DDM_M2P_JAX(src_v, src_f, tgt_points, K=5, up_ratio=3, std=0.05, beta=0)`
- `src_v`: (N, 3) jax array — mesh vertices
- `src_f`: (F, 3) int array — mesh face indices
- `tgt_points`: (M, 3) jax array — target point cloud
- Returns: scalar loss

Port the logic from `DDM_M2P.forward()` in `DDM/__init__.py`. Key operations:
1. Compute face centers from `src_v[src_f]`
2. Generate noisy query points around `tgt_points`, concatenate with face centers
3. For each query point, find the closest point on the mesh surface — this is the hard part (see below)
4. Compute `dir_src = query - closest_src`, `udf_src = norm(dir_src)`
5. For each query point, find K nearest neighbors in `tgt_points`, compute inverse-distance-weighted UDF gradient → `dir_tgt`, `udf_tgt`
6. Compute `errors = sum(|geo_src - geo_tgt|, dim=-1)` where `geo = cat([dir, udf], dim=-1)`
7. Weight by `exp(-errors * beta)` (stop-gradient), return weighted mean

### The `closestPointOnSurface` problem

The PyTorch version uses a custom CUDA kernel (`closest_point_on_surface.cu`) that finds, for each query point, the closest point on a triangle mesh. It returns the face index and barycentric weights (w1, w2, w3).

The CUDA kernel iterates over ALL faces for each query point (brute force) and uses the standard closest-point-on-triangle algorithm (project onto triangle plane, then clamp to triangle edges/vertices using the 7-region Voronoi test on parameters s, t).

For JAX, implement this as a pure JAX function:
- `closest_point_on_triangles(points, f1, f2, f3)` → `(indices, w1, w2, w3)`
- `points`: (Q, 3), `f1/f2/f3`: (F, 3) — triangle vertices
- For each point, compute distance to ALL triangles, return the closest
- The per-triangle closest point uses barycentric coordinates: given point P and triangle (A, B, C), compute `s, t` parameterizing the closest point as `A + s*(B-A) + t*(C-A)`, with the 7-region clamping logic from the CUDA kernel
- This can be vectorized: compute `(Q, F)` distance matrix, then `argmin` over faces

**Important**: The brute-force all-pairs approach may be memory-intensive for large meshes. For our use case, meshes are ~20K faces and query point counts are moderate (~10K-60K), so `(Q, F)` should fit in GPU memory. If not, chunk over query points.

## Where to put the code

Add a new file `DDM/jax_ddm.py` with the implementations. Keep the existing PyTorch code untouched.

## Testing

After implementing, add a test at the bottom of `jax_ddm.py` (guarded by `if __name__ == "__main__"`) that:
1. Creates random mesh vertices/faces and point clouds
2. Runs both the PyTorch and JAX versions on the same data
3. Asserts the outputs are close (within reasonable tolerance for float32)
4. Times both versions
5. Tests that `jax.jit(ddm_m2p_jax)(...)` works
6. Tests that `jax.vmap` over a batch of source vertices works (simulating vmapping over candidate poses)

## Key JAX considerations

- All operations must be JIT-compatible — no Python control flow on traced values, no dynamic shapes
- The closest-point-on-triangle clamping has many branches — use `jnp.where` chains instead of if/else
- Random noise generation should accept a `jax.random.PRNGKey` argument instead of using `torch.randn`
- `stop_gradient` in JAX is `jax.lax.stop_gradient`
- For KNN, avoid materializing the full `(B, N, M)` distance matrix if M is large — but for our scale (~500 points) it's fine
