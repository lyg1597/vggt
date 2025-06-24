import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def solve_T_s(a: np.ndarray, b: np.ndarray,
              init_s: float | None = None) -> tuple[np.ndarray, float]:
    """
    Estimate T (4×4) and scalar s such that T @ (a with scaled translation) ≈ b.

    Parameters
    ----------
    a, b : ndarray, shape (N, 4, 4)
        Homogeneous transforms for two coordinate systems.
    init_s : float, optional
        Initial guess for the scale (defaults to 1.0 or the ratio of median
        translation norms).

    Returns
    -------
    T_hat : ndarray, shape (4, 4)
        Estimated similarity transform.
    s_hat : float
        Estimated translation scale factor.
    """
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.shape[1:] == (4, 4)
    N = a.shape[0]

    # Extract rotations and translations
    Ra = a[:, :3, :3]
    ta = a[:, :3, 3]
    Rb = b[:, :3, :3]
    tb = b[:, :3, 3]

    # ---------- initial guesses ------------------------------------------------
    # Rotation: average of R_b R_aᵀ (projected back to SO(3))
    R_init = Rotation.from_matrix(Rb @ np.transpose(Ra, (0, 2, 1))).mean()
    r_init = R_init.as_rotvec()
    # Translation: rough by aligning centroids (with s ≈ 1)
    if init_s is None:
        init_s = np.median(np.linalg.norm(tb, axis=1) /
                           np.maximum(np.linalg.norm(ta, axis=1), 1e-9))
    t_init = tb.mean(axis=0) - R_init.apply(init_s * ta.mean(axis=0))

    # Parameter vector p = [rx, ry, rz, tx, ty, tz, s]
    p0 = np.hstack([r_init, t_init, init_s])

    # ---------- residual --------------------------------------------------------
    def residual(p):
        r_vec, t_vec, s = p[:3], p[3:6], p[6]
        R = Rotation.from_rotvec(r_vec).as_matrix()
        # orientation residual: flatten matrices
        R_pred = R @ Ra                        # (N, 3, 3)
        orient_res = (R_pred - Rb).reshape(N, -1)
        # translation residual
        trans_pred = t_vec + (R @ (ta.T * s)).T
        trans_res = trans_pred - tb
        return np.concatenate([orient_res, trans_res], axis=1).ravel()

    # ---------- solve -----------------------------------------------------------
    res = least_squares(residual, p0, method="lm")   # Levenberg–Marquardt

    r_opt, t_opt, s_opt = res.x[:3], res.x[3:6], res.x[6]
    R_opt = Rotation.from_rotvec(r_opt).as_matrix()

    T_opt = np.eye(4)
    T_opt[:3, :3] = R_opt
    T_opt[:3, 3] = t_opt
    return T_opt, s_opt

# --- assume you already have -----------------------------------------------
# a, b               # (N, 4, 4) arrays of homogeneous transforms
# T_hat, s_hat       # returned by solve_T_s(a, b)
# ---------------------------------------------------------------------------

def a_to_b(a, T, s):
    """
    Apply T and scale s to take poses in frame A → frame B.

    b_pred_i = T @ [ R_a_i | s * t_a_i ]
    """
    a_scaled = a.copy()
    a_scaled[:, :3, 3] *= s               # scale the translation only
    return (T @ a_scaled)                 # broadcasted matrix product

def b_to_a(b, T, s):
    """
    Inverse mapping: recover A-frame poses from B-frame ones.

    a_pred_i = inv(T) @ b_i;  then un-scale the translation.
    """
    T_inv = np.linalg.inv(T)
    a_pred = T_inv @ b
    a_pred[:, :3, 3] /= s                 # undo the scale on translation
    return a_pred

# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    b_pred = a_to_b(a, T_hat, s_hat)          # reconstruct b from a
    a_pred = b_to_a(b, T_hat, s_hat)          # reconstruct a from b

    # quick sanity-check
    rot_err  = np.max(np.linalg.norm(b_pred[:, :3, :3] - b[:, :3, :3], axis=(1, 2)))
    trans_err = np.max(np.linalg.norm(b_pred[:, :3, 3] - b[:, :3, 3], axis=1))
    print(f"max orientation error: {rot_err:.3e}")
    print(f"max translation error: {trans_err:.3e}")

    
