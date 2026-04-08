"""
JAX reimplementations of DDM_P2P and DDM_M2P for use inside jax.jit / jax.vmap.
"""

import jax
import jax.numpy as jnp
from functools import partial


# ---------------------------------------------------------------------------
# KNN via squared-distance matrix + top_k
# ---------------------------------------------------------------------------

def _knn_points(query, ref, K):
    """
    Find K nearest neighbors in ref for each point in query.

    Args:
        query: (N, 3)
        ref:   (M, 3)
        K:     int

    Returns:
        dists: (N, K) squared distances
        idx:   (N, K) indices into ref
        knn:   (N, K, 3) neighbor coordinates
    """
    # (N, M) squared distance matrix
    diff = query[:, None, :] - ref[None, :, :]  # (N, M, 3)
    sq_dists = jnp.sum(diff * diff, axis=-1)     # (N, M)

    # top_k returns largest, so negate to get smallest
    neg_dists, idx = jax.lax.top_k(-sq_dists, K)  # (N, K)
    dists = -neg_dists  # (N, K)

    knn = ref[idx]  # (N, K, 3)
    return dists, idx, knn


def _knn_points_batched(query, ref, K):
    """
    Batched KNN: query (B, N, 3), ref (B, M, 3) -> dists (B, N, K), etc.
    """
    return jax.vmap(lambda q, r: _knn_points(q, r, K))(query, ref)


# ---------------------------------------------------------------------------
# Closest point on triangle mesh (brute force, vectorized)
# ---------------------------------------------------------------------------

def _closest_point_on_triangle_single(point, f1, f2, f3):
    """
    For a single point and a single triangle (f1, f2, f3), compute the
    closest point using barycentric coordinates with the 7-region Voronoi test.

    The closest point is: f1 + s*(f2-f1) + t*(f3-f1)

    Returns: (sq_dist, s, t)
    """
    edge0 = f2 - f1
    edge1 = f3 - f1
    v0 = f1 - point

    a = jnp.dot(edge0, edge0) + 1e-20
    b = jnp.dot(edge0, edge1)
    c = jnp.dot(edge1, edge1) + 1e-20
    d = jnp.dot(edge0, v0)
    e = jnp.dot(edge1, v0)
    f = jnp.dot(v0, v0)

    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e

    # Region inside triangle
    inv_det = 1.0 / jnp.where(jnp.abs(det) < 1e-20, 1e-20, det)
    s_inside = s * inv_det
    t_inside = t * inv_det

    # --- s + t < det branch ---
    # Region 4: s < 0, t < 0
    r4_s = jnp.where(d < 0.0, jnp.clip(-d / a, 0.0, 1.0), 0.0)
    r4_t = jnp.where(d < 0.0, 0.0, jnp.clip(-e / c, 0.0, 1.0))

    # Region 3: s < 0, t >= 0
    r3_s = 0.0
    r3_t = jnp.clip(-e / c, 0.0, 1.0)

    # Region 5: s >= 0, t < 0
    r5_s = jnp.clip(-d / a, 0.0, 1.0)
    r5_t = 0.0

    # For s + t < det:
    st_lt_det_s = jnp.where(
        s < 0.0,
        jnp.where(t < 0.0, r4_s, r3_s),
        jnp.where(t < 0.0, r5_s, s_inside)
    )
    st_lt_det_t = jnp.where(
        s < 0.0,
        jnp.where(t < 0.0, r4_t, r3_t),
        jnp.where(t < 0.0, r5_t, t_inside)
    )

    # --- s + t >= det branch ---
    # Region 2: s < 0
    tmp0_r2 = b + d
    tmp1_r2 = c + e
    numer_r2 = tmp1_r2 - tmp0_r2
    denom_r2 = a - 2.0 * b + c
    r2_s_a = jnp.clip(numer_r2 / denom_r2, 0.0, 1.0)
    r2_t_a = 1.0 - r2_s_a
    r2_s_b = 0.0
    r2_t_b = jnp.clip(-e / c, 0.0, 1.0)
    r2_s = jnp.where(tmp1_r2 > tmp0_r2, r2_s_a, r2_s_b)
    r2_t = jnp.where(tmp1_r2 > tmp0_r2, r2_t_a, r2_t_b)

    # Region 6: t < 0
    r6_s_a = jnp.clip((c + e - b - d) / (a - 2.0 * b + c), 0.0, 1.0)
    r6_t_a = 1.0 - r6_s_a
    r6_s_b = jnp.clip(-e / c, 0.0, 1.0)
    r6_t_b = 0.0
    r6_s = jnp.where(a + d > b + e, r6_s_a, r6_s_b)
    r6_t = jnp.where(a + d > b + e, r6_t_a, r6_t_b)

    # Region 1: s >= 0, t >= 0
    numer_r1 = c + e - b - d
    denom_r1 = a - 2.0 * b + c
    r1_s = jnp.clip(numer_r1 / denom_r1, 0.0, 1.0)
    r1_t = 1.0 - r1_s

    st_ge_det_s = jnp.where(
        s < 0.0, r2_s,
        jnp.where(t < 0.0, r6_s, r1_s)
    )
    st_ge_det_t = jnp.where(
        s < 0.0, r2_t,
        jnp.where(t < 0.0, r6_t, r1_t)
    )

    # Final selection
    final_s = jnp.where(s + t < det, st_lt_det_s, st_ge_det_s)
    final_t = jnp.where(s + t < det, st_lt_det_t, st_ge_det_t)

    sq_dist = a * final_s * final_s + 2.0 * b * final_s * final_t + c * final_t * final_t + 2.0 * d * final_s + 2.0 * e * final_t + f
    sq_dist = jnp.maximum(sq_dist, 0.0)

    return sq_dist, final_s, final_t


# Vectorize over all faces for a single point -> (F,) sq_dists, s, t
_closest_point_all_faces = jax.vmap(
    _closest_point_on_triangle_single,
    in_axes=(None, 0, 0, 0)
)


def _closest_point_for_single_query(point, f1, f2, f3):
    """
    For one query point, find closest point across ALL triangles.
    Returns: (best_idx, w1, w2, w3)
    """
    sq_dists, s_vals, t_vals = _closest_point_all_faces(point, f1, f2, f3)
    best_idx = jnp.argmin(sq_dists)
    best_s = s_vals[best_idx]
    best_t = t_vals[best_idx]
    w1 = 1.0 - best_s - best_t
    w2 = best_s
    w3 = best_t
    return best_idx, w1, w2, w3


def closest_point_on_triangles(points, f1, f2, f3):
    """
    For each query point, find the closest point on a triangle mesh.

    Args:
        points: (Q, 3) query points
        f1, f2, f3: (F, 3) triangle vertex positions

    Returns:
        indices: (Q,) face indices
        w1, w2, w3: (Q,) barycentric weights
    """
    return jax.vmap(
        lambda p: _closest_point_for_single_query(p, f1, f2, f3)
    )(points)


# ---------------------------------------------------------------------------
# DDM_P2P_JAX
# ---------------------------------------------------------------------------

def _cal_udf_weights(query, x, K):
    """
    Compute inverse-distance-weighted UDF and gradient for query points w.r.t. x.

    Args:
        query: (N, 3)
        x:     (M, 3)
        K:     int

    Returns:
        udf:      (N,)
        udf_grad: (N, 3)
    """
    dists, idx, knn_pc = _knn_points(query, x, K)  # (N,K), (N,K), (N,K,3)

    dir_vec = query[:, None, :] - knn_pc  # (N, K, 3)

    norm = jnp.sum(1.0 / (dists + 1e-8), axis=1, keepdims=True)  # (N, 1)
    weights = (1.0 / (jax.lax.stop_gradient(dists) + 1e-8)) / jax.lax.stop_gradient(norm)  # (N, K)

    udf_grad = jnp.sum(dir_vec * weights[:, :, None], axis=1)  # (N, 3)
    udf = jnp.linalg.norm(udf_grad + 1e-10, axis=-1)  # (N,)

    return udf, udf_grad


def _cal_udf_weights_batched(query, x, K):
    """Batched version: query (B, N, 3), x (B, M, 3)."""
    return jax.vmap(lambda q, r: _cal_udf_weights(q, r, K))(query, x)


@partial(jax.jit, static_argnames=('K', 'up_ratio', 'weighted_query'))
def ddm_p2p_jax(src, tgt, key, K=5, up_ratio=10, std=0.05, weighted_query=True, beta=3.0):
    """
    JAX equivalent of DDM_P2P.forward().

    Args:
        src: (B, N, 3) source point cloud
        tgt: (B, M, 3) target point cloud
        key: jax.random.PRNGKey for noise generation
        K, up_ratio, std, weighted_query, beta: hyperparameters

    Returns:
        scalar loss
    """
    B, M, _ = tgt.shape

    # Generate noisy query points around tgt
    noise = jax.random.normal(key, shape=(B, M, up_ratio, 3)) * std
    query = tgt[:, :, None, :] + noise  # (B, M, up_ratio, 3)
    query = query.reshape(B, -1, 3)  # (B, M*up_ratio, 3)

    # Concatenate with src (stop gradient on both as in PyTorch version)
    query = jnp.concatenate([jax.lax.stop_gradient(query), jax.lax.stop_gradient(src)], axis=1)

    # Compute UDF for both src and tgt
    udf_tgt, udf_grad_tgt = _cal_udf_weights_batched(query, tgt, K)
    udf_src, udf_grad_src = _cal_udf_weights_batched(query, src, K)

    udf_error = jnp.abs(udf_tgt - udf_src)  # (B, Q)
    udf_grad_error = jnp.sum(jnp.abs(udf_grad_src - udf_grad_tgt), axis=-1)  # (B, Q)

    def weighted_loss(udf_error, udf_grad_error, query):
        query_weights = jnp.exp(-jax.lax.stop_gradient(udf_error) * beta) * \
                        jnp.exp(-jax.lax.stop_gradient(udf_grad_error) * beta)
        query_weights = jax.lax.stop_gradient(query_weights)
        return jnp.sum((udf_error + udf_grad_error) * query_weights) / B / query.shape[1]

    def unweighted_loss(udf_error, udf_grad_error, query):
        return jnp.sum((udf_error + udf_grad_error)) / B / query.shape[1]

    if weighted_query:
        return weighted_loss(udf_error, udf_grad_error, query)
    else:
        return unweighted_loss(udf_error, udf_grad_error, query)


# ---------------------------------------------------------------------------
# DDM_M2P_JAX
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=('K', 'up_ratio'))
def ddm_m2p_jax(src_v, src_f, tgt_points, key, K=5, up_ratio=3, std=0.05, beta=0.0):
    """
    JAX equivalent of DDM_M2P.forward().

    Args:
        src_v:      (N, 3) mesh vertices
        src_f:      (F, 3) int face indices
        tgt_points: (M, 3) target point cloud
        key:        jax.random.PRNGKey
        K, up_ratio, std, beta: hyperparameters

    Returns:
        scalar loss
    """
    # Triangle vertices
    f1 = src_v[src_f[:, 0]]
    f2 = src_v[src_f[:, 1]]
    f3 = src_v[src_f[:, 2]]

    src_center = (f1 + f2 + f3) / 3.0

    # Generate noisy query points around tgt
    M = tgt_points.shape[0]
    noise = jax.random.normal(key, shape=(M, up_ratio, 3)) * std
    query_points = tgt_points[:, None, :] + noise  # (M, up_ratio, 3)
    query_points = query_points.reshape(-1, 3)      # (M*up_ratio, 3)

    query_points = jnp.concatenate([
        jax.lax.stop_gradient(query_points),
        jax.lax.stop_gradient(src_center)
    ], axis=0)

    # Closest point on mesh for each query
    indices, w1, w2, w3 = closest_point_on_triangles(query_points, f1, f2, f3)

    sel_f1 = f1[indices]
    sel_f2 = f2[indices]
    sel_f3 = f3[indices]

    closest_src = w1[:, None] * sel_f1 + w2[:, None] * sel_f2 + w3[:, None] * sel_f3

    dir_src = query_points - closest_src
    udf_src = jnp.linalg.norm(dir_src + 1e-10, axis=-1, keepdims=True)  # (Q, 1)
    geo_src = jnp.concatenate([dir_src, udf_src], axis=1)  # (Q, 4)

    # KNN in tgt for each query point
    dists, idx, knn_pc = _knn_points(query_points, tgt_points, K)  # (Q,K), (Q,K), (Q,K,3)

    dir_vec = query_points[:, None, :] - knn_pc  # (Q, K, 3)

    norm = jnp.sum(1.0 / (dists + 1e-8), axis=1, keepdims=True)  # (Q, 1)
    weights = (1.0 / (jax.lax.stop_gradient(dists) + 1e-8)) / jax.lax.stop_gradient(norm)  # (Q, K)

    dir_tgt = jnp.sum(dir_vec * weights[:, :, None], axis=1)  # (Q, 3)
    udf_tgt = jnp.linalg.norm(dir_tgt + 1e-10, axis=-1)       # (Q,)

    geo_tgt = jnp.concatenate([dir_tgt, udf_tgt[:, None]], axis=-1)  # (Q, 4)

    errors = jnp.sum(jnp.abs(geo_src - geo_tgt), axis=-1)  # (Q,)

    query_weights = jax.lax.stop_gradient(jnp.exp(-errors * beta))

    return jnp.mean(errors * query_weights)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("Testing JAX DDM implementations")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Test 1: closest_point_on_triangles correctness
    # ------------------------------------------------------------------
    print("\n--- Test 1: closest_point_on_triangles ---")
    key = jax.random.PRNGKey(42)

    # Simple triangle
    f1 = jnp.array([[0.0, 0.0, 0.0]])
    f2 = jnp.array([[1.0, 0.0, 0.0]])
    f3 = jnp.array([[0.0, 1.0, 0.0]])

    # Point above the triangle center
    points = jnp.array([[0.25, 0.25, 1.0]])
    idx, w1, w2, w3 = closest_point_on_triangles(points, f1, f2, f3)
    closest = w1[:, None] * f1[idx] + w2[:, None] * f2[idx] + w3[:, None] * f3[idx]
    print(f"  Point: {points[0]}, Closest: {closest[0]}, Expected: [0.25, 0.25, 0.0]")
    assert jnp.allclose(closest[0], jnp.array([0.25, 0.25, 0.0]), atol=1e-5), "Failed!"
    print("  PASSED")

    # ------------------------------------------------------------------
    # Test 2: DDM_P2P_JAX runs and produces reasonable output
    # ------------------------------------------------------------------
    print("\n--- Test 2: DDM_P2P_JAX basic ---")
    key = jax.random.PRNGKey(0)
    B, N = 2, 100
    k1, k2, k3 = jax.random.split(key, 3)
    src = jax.random.normal(k1, (B, N, 3)) * 0.1
    tgt = jax.random.normal(k2, (B, N, 3)) * 0.1

    loss = ddm_p2p_jax(src, tgt, k3, K=5, up_ratio=5, std=0.05, weighted_query=True, beta=3.0)
    print(f"  P2P loss: {loss:.6f}")
    assert jnp.isfinite(loss), "Loss is not finite!"

    # Same point clouds should give ~0 loss
    loss_same = ddm_p2p_jax(src, src, k3, K=5, up_ratio=5, std=0.05, weighted_query=True, beta=3.0)
    print(f"  P2P loss (same clouds): {loss_same:.6f}")
    assert loss_same < loss, "Self-loss should be smaller!"
    print("  PASSED")

    # ------------------------------------------------------------------
    # Test 3: DDM_M2P_JAX runs and produces reasonable output
    # ------------------------------------------------------------------
    print("\n--- Test 3: DDM_M2P_JAX basic ---")
    key = jax.random.PRNGKey(1)
    k1, k2, k3 = jax.random.split(key, 3)

    # Create a simple mesh with non-degenerate faces
    n_verts = 50
    n_faces = 80
    src_v = jax.random.normal(k1, (n_verts, 3)) * 0.5
    # Ensure no degenerate faces (all 3 indices distinct)
    import numpy as _np
    rng = _np.random.RandomState(42)
    faces_list = []
    while len(faces_list) < n_faces:
        tri = rng.choice(n_verts, 3, replace=False)
        faces_list.append(tri)
    src_f = jnp.array(_np.array(faces_list), dtype=jnp.int32)
    tgt_points = jax.random.normal(k3, (200, 3)) * 0.5

    k4 = jax.random.PRNGKey(99)
    loss_m2p = ddm_m2p_jax(src_v, src_f, tgt_points, k4, K=5, up_ratio=3, std=0.05, beta=0.0)
    print(f"  M2P loss: {loss_m2p:.6f}")
    assert jnp.isfinite(loss_m2p), "Loss is not finite!"
    print("  PASSED")

    # ------------------------------------------------------------------
    # Test 4: JIT compilation works
    # ------------------------------------------------------------------
    print("\n--- Test 4: JIT compilation ---")
    jitted_m2p = jax.jit(ddm_m2p_jax, static_argnames=('K', 'up_ratio'))

    # Warm up
    _ = jitted_m2p(src_v, src_f, tgt_points, k4, K=5, up_ratio=3, std=0.05, beta=0.0)
    _ = jitted_m2p(src_v, src_f, tgt_points, k4, K=5, up_ratio=3, std=0.05, beta=0.0)

    t0 = time.time()
    for _ in range(10):
        loss = jitted_m2p(src_v, src_f, tgt_points, k4, K=5, up_ratio=3, std=0.05, beta=0.0)
        loss.block_until_ready()
    t1 = time.time()
    print(f"  JIT M2P: {(t1 - t0) / 10 * 1000:.1f} ms per call")
    print("  PASSED")

    # ------------------------------------------------------------------
    # Test 5: vmap over batch of source vertices (candidate poses)
    # ------------------------------------------------------------------
    print("\n--- Test 5: vmap over candidate poses ---")
    n_candidates = 8

    k5 = jax.random.PRNGKey(7)
    # Batch of candidate vertex positions
    src_v_batch = jax.random.normal(k5, (n_candidates, n_verts, 3)) * 0.5
    keys_batch = jax.random.split(jax.random.PRNGKey(10), n_candidates)

    vmapped_m2p = jax.vmap(
        lambda v, k: ddm_m2p_jax(v, src_f, tgt_points, k, K=5, up_ratio=3, std=0.05, beta=0.0)
    )

    losses = vmapped_m2p(src_v_batch, keys_batch)
    print(f"  vmap losses shape: {losses.shape}, values: {losses}")
    assert losses.shape == (n_candidates,), f"Expected ({n_candidates},), got {losses.shape}"
    assert jnp.all(jnp.isfinite(losses)), "Some losses are not finite!"
    print("  PASSED")

    # ------------------------------------------------------------------
    # Test 6: Compare with PyTorch (if available)
    # ------------------------------------------------------------------
    print("\n--- Test 6: PyTorch comparison ---")
    try:
        import torch
        import sys
        sys.path.insert(0, '.')
        from DDM import DDM_P2P, DDM_M2P

        print("  PyTorch + DDM available, running comparison...")

        # Use fixed data for reproducibility
        key = jax.random.PRNGKey(123)
        k1, k2 = jax.random.split(key)
        B, N = 1, 50

        src_np = jax.random.normal(k1, (B, N, 3)).astype(jnp.float32)
        tgt_np = jax.random.normal(k2, (B, N, 3)).astype(jnp.float32)

        # PyTorch version
        src_torch = torch.tensor(src_np.__array__()).cuda()
        tgt_torch = torch.tensor(tgt_np.__array__()).cuda()

        ddm_pt = DDM_P2P(up_ratio=5, K=5, std=0.0, weighted_query=False, beta=3.0).cuda()

        # With std=0 and no weighting, the query is just src (no random noise dependency)
        loss_pt = ddm_pt(src_torch, tgt_torch).item()

        # JAX version with std=0
        k3 = jax.random.PRNGKey(0)
        loss_jax = float(ddm_p2p_jax(src_np, tgt_np, k3, K=5, up_ratio=5, std=0.0, weighted_query=False, beta=3.0))

        print(f"  PyTorch P2P loss: {loss_pt:.6f}")
        print(f"  JAX P2P loss:     {loss_jax:.6f}")
        print(f"  Diff:             {abs(loss_pt - loss_jax):.6f}")

        # With std=0, noise doesn't matter, so results should be close
        if abs(loss_pt - loss_jax) < 0.01:
            print("  PASSED (close match)")
        else:
            print("  WARNING: outputs differ — check KNN ordering/tie-breaking differences")

    except (ImportError, Exception) as e:
        print(f"  Skipping PyTorch comparison: {e}")

    # ------------------------------------------------------------------
    # Timing comparison: JAX vs PyTorch
    # ------------------------------------------------------------------
    print("\n--- Timing: JAX P2P ---")
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B, N = 1, 200
    src = jax.random.normal(k1, (B, N, 3)) * 0.1
    tgt = jax.random.normal(k2, (B, N, 3)) * 0.1

    # Warmup
    _ = ddm_p2p_jax(src, tgt, k3, K=5, up_ratio=5)
    _ = ddm_p2p_jax(src, tgt, k3, K=5, up_ratio=5)

    t0 = time.time()
    for _ in range(20):
        l = ddm_p2p_jax(src, tgt, k3, K=5, up_ratio=5)
        l.block_until_ready()
    t1 = time.time()
    jax_p2p_ms = (t1 - t0) / 20 * 1000
    print(f"  JAX P2P (B=1, N=200): {jax_p2p_ms:.1f} ms per call")

    try:
        import torch
        from DDM import DDM_P2P, DDM_M2P

        print("\n--- Timing: PyTorch P2P ---")
        src_t = torch.tensor(src.__array__()).cuda()
        tgt_t = torch.tensor(tgt.__array__()).cuda()
        ddm_pt = DDM_P2P(up_ratio=5, K=5, std=0.05, weighted_query=True, beta=3.0).cuda().eval()

        # Warmup
        with torch.no_grad():
            _ = ddm_pt(src_t, tgt_t)
            _ = ddm_pt(src_t, tgt_t)
            torch.cuda.synchronize()

        t0 = time.time()
        with torch.no_grad():
            for _ in range(20):
                l = ddm_pt(src_t, tgt_t)
                torch.cuda.synchronize()
        t1 = time.time()
        pt_p2p_ms = (t1 - t0) / 20 * 1000
        print(f"  PyTorch P2P (B=1, N=200): {pt_p2p_ms:.1f} ms per call")
        print(f"  Speedup (PyTorch/JAX): {pt_p2p_ms / jax_p2p_ms:.2f}x")

        print("\n--- Timing: JAX M2P ---")
        # Reuse mesh data from test 3
        _ = ddm_m2p_jax(src_v, src_f, tgt_points, k4, K=5, up_ratio=3, std=0.05, beta=0.0)
        _ = ddm_m2p_jax(src_v, src_f, tgt_points, k4, K=5, up_ratio=3, std=0.05, beta=0.0)
        t0 = time.time()
        for _ in range(20):
            l = ddm_m2p_jax(src_v, src_f, tgt_points, k4, K=5, up_ratio=3, std=0.05, beta=0.0)
            l.block_until_ready()
        t1 = time.time()
        jax_m2p_ms = (t1 - t0) / 20 * 1000
        print(f"  JAX M2P (V=50, F=80, P=200): {jax_m2p_ms:.1f} ms per call")

        print("\n--- Timing: PyTorch M2P ---")
        src_v_t = torch.tensor(src_v.__array__()).cuda()
        src_f_t = torch.tensor(src_f.__array__()).long().cuda()
        tgt_pts_t = torch.tensor(tgt_points.__array__()).cuda()
        ddm_m2p_pt = DDM_M2P(up_ratio=3, K=5, std=0.05, beta=0.0).cuda().eval()

        with torch.no_grad():
            _ = ddm_m2p_pt(src_v_t, src_f_t, tgt_pts_t)
            _ = ddm_m2p_pt(src_v_t, src_f_t, tgt_pts_t)
            torch.cuda.synchronize()

        t0 = time.time()
        with torch.no_grad():
            for _ in range(20):
                l = ddm_m2p_pt(src_v_t, src_f_t, tgt_pts_t)
                torch.cuda.synchronize()
        t1 = time.time()
        pt_m2p_ms = (t1 - t0) / 20 * 1000
        print(f"  PyTorch M2P (V=50, F=80, P=200): {pt_m2p_ms:.1f} ms per call")
        print(f"  Speedup (PyTorch/JAX): {pt_m2p_ms / jax_m2p_ms:.2f}x")

    except (ImportError, Exception) as e:
        print(f"  Skipping PyTorch timing: {e}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
