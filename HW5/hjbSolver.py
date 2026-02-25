# %%
"""
Homework: Capacity Expansion Strategy under Different Market Scenarios

You are a consultant advising a power generation consortium on optimal 
capacity expansion strategy. The consortium can invest in multiple energy 
technologies with different characteristics.

Part 1: Base Case Analysis
Part 2: Technology Comparison
"""

import numpy as np
from scipy.sparse import spdiags, eye, kron, lil_matrix, csr_matrix
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# %%
def price(s, x, s_eps=1e-12):
    """
    Price function P(s, x) = sqrt(x / max(s, s_eps)) - 1.
    """
    return np.sqrt(x / np.maximum(s, s_eps)) - 1.0


# %%
def build_dx_operator(J, I):
    """
    Build sparse Ax operator corresponding to forward difference in x.
    """
    Dx = lil_matrix((J + 1, J + 1))
    for jj in range(J):
        Dx[jj, jj] = -1.0
        Dx[jj, jj + 1] = 1.0

    Dx = Dx.tocsr()
    Is = eye(I + 1, format="csr")
    Ax = kron(Dx, Is, format="csr")
    return Ax


# %%
def build_ds_operators(delta_vec, ds, I, J):
    """
    Build the list of As operators, one for each technology.
    """
    dtech = len(delta_vec)
    As_list = []
    for t in range(dtech):
        delta = delta_vec[t]
        k = int(round(delta / ds))
        Ds = lil_matrix((I + 1, I + 1))

        if k > 0:
            last_row = I - k + 1
            if last_row >= 1:
                for ii in range(last_row):
                    Ds[ii, ii] = -1.0
                    Ds[ii, ii + k] = 1.0
        elif k < 0:
            kk = -k
            first_row = kk
            if first_row <= I:
                for ii in range(first_row, I + 1):
                    Ds[ii, ii] = -1.0
                    Ds[ii, ii + k] = 1.0
        else:
            pass

        Ds = Ds.tocsr()
        Ix = eye(J + 1, format="csr")
        As = kron(Ix, Ds, format="csr")
        As_list.append(As)

    return As_list


# %%
def multi_rhs(t, V, As_list, Ax, Rvec, r, lambdaX, beta_vec, rho_vec, expo_vec):
    """
    Right-hand side of the HJB ODE system.
    """
    V = V.reshape(-1, 1)
    Ntech = len(As_list)
    Phi_sum = np.zeros_like(V)

    for j in range(Ntech):
        AsV = As_list[j].dot(V)
        tmp = AsV - rho_vec[j]
        lam = np.zeros_like(tmp)

        mask = tmp > 0.0
        lam[mask] = np.power(tmp[mask], expo_vec[j])

        Phi_j = np.zeros_like(tmp)
        if np.any(mask):
            lam_pos = lam[mask]
            z_pos = AsV[mask]
            beta_j = beta_vec[j]
            Phi_j[mask] = lam_pos * z_pos \
                          - (1.0 / beta_j) * np.power(lam_pos, beta_j) \
                          - rho_vec[j] * lam_pos

        Phi_sum += Phi_j

    dV = -Phi_sum - lambdaX * (Ax.dot(V)) - Rvec.reshape(-1, 1) + r * V
    return dV.ravel()


# %%
def solve_hjb(T, r, lambdaX, dx, ds, s_min, I, J, delta_vec, beta_vec, rho_vec, 
              tech_names=None, s_eps=1e-12, verbose=True):
    """
    Solve the HJB equation and return comprehensive results.
    """
    if verbose:
        print(f"  Solving HJB: T={T}, r={r}, λ_X={lambdaX}")
        print(f"  Grid: I={I}, J={J}, ds={ds}, dx={dx}")
    
    # Grids
    S = s_min + ds * np.arange(I + 1)
    X = dx * np.arange(J + 1)
    Sgrid, Xgrid = np.meshgrid(S, X, indexing="ij")

    dtech = len(delta_vec)
    if tech_names is None:
        tech_names = [f"Tech_{i+1}" for i in range(dtech)]

    # Sanity checks
    k_stride = np.round(delta_vec / ds).astype(int)
    if np.any(np.abs(k_stride * ds - delta_vec) > 1e-12):
        raise ValueError("Each delta_vec[j] must be integer multiple of ds.")
    if np.any(beta_vec <= 1.0):
        raise ValueError("All beta_vec[j] must be > 1.")

    expo_vec = 1.0 / (beta_vec - 1.0)

    # Reward
    Rmat = Sgrid * price(Sgrid, Xgrid, s_eps=s_eps)
    Rvec = Rmat.reshape(-1, order="F")
    N = Rvec.size

    # Operators
    Ax = build_dx_operator(J, I)
    As_list = build_ds_operators(delta_vec, ds, I, J)

    # RHS wrapper
    def rhs(t, V):
        return multi_rhs(t, V, As_list, Ax, Rvec, r, lambdaX, beta_vec, rho_vec, expo_vec)

    # Terminal condition
    V_T = np.zeros(N)

    # Integrate backward in time
    if verbose:
        print("  Integrating HJB backward in time...")
    
    sol = solve_ivp(
        rhs,
        t_span=(T, 0.0),
        y0=V_T,
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solve failed: {sol.message}")

    if verbose:
        print("  ✓ Integration successful")

    # V(0)
    V0_vec = sol.y[:, -1]
    V0 = V0_vec.reshape((I + 1, J + 1), order="F")

    # Compute λ*_j(t=0, s, x) and marginal values
    LambdaStar0_j = np.zeros((I + 1, J + 1, dtech))
    LambdaStar0_sum = np.zeros((I + 1, J + 1))
    DeltaS_v = np.zeros((I + 1, J + 1, dtech))

    for idx_tech in range(dtech):
        k = k_stride[idx_tech]
        Dsv0 = np.zeros_like(V0)

        if k > 0:
            last_row = I - k + 1
            if last_row >= 1:
                Dsv0[0:last_row, :] = V0[k:k + last_row, :] - V0[0:last_row, :]
        elif k < 0:
            kk = -k
            first_row = kk
            if first_row <= I:
                Dsv0[first_row:I + 1, :] = V0[0:I + 1 - kk, :] - V0[first_row:I + 1, :]

        DeltaS_v[:, :, idx_tech] = Dsv0

        tmp = Dsv0 - rho_vec[idx_tech]
        lam = np.zeros_like(tmp)
        mask = tmp > 0.0
        lam[mask] = np.power(tmp[mask], expo_vec[idx_tech])

        LambdaStar0_j[:, :, idx_tech] = lam
        LambdaStar0_sum += lam

    return {
        "S": S,
        "X": X,
        "Sgrid": Sgrid,
        "Xgrid": Xgrid,
        "V0": V0,
        "LambdaStar0_j": LambdaStar0_j,
        "LambdaStar0_sum": LambdaStar0_sum,
        "DeltaS_v": DeltaS_v,
        "price_grid": price(Sgrid, Xgrid, s_eps=s_eps),
        "tech_names": tech_names,
        "params": {
            "T": T, "r": r, "lambdaX": lambdaX,
            "ds": ds, "dx": dx, "I": I, "J": J,
            "delta_vec": delta_vec,
            "beta_vec": beta_vec,
            "rho_vec": rho_vec,
        }
    }