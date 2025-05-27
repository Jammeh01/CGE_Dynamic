import scipy.optimize as opt
import numpy as np
import pandas as pd
from pandas import Series
import os
from open_cge import government as gov
from open_cge import household as hh
from open_cge import aggregates as agg
from open_cge import firms, calibrate
from open_cge import simpleCGE as cge

# --- Load Social Accounting Matrix ---
current_path = os.path.abspath(os.path.dirname(__file__))
sam_path = os.path.join(current_path, "SAM.xlsx")
sam = pd.read_excel(sam_path, index_col=0, header=0)

# --- Variable Sets ---
u = (
    "AGR", "OIL", "IND", "SER", "LAB", "CAP", "LAND", "NTR",
    "DTX", "IDT", "ACT", "HOH", "GOV", "INV", "EXT",
)
ind = ("AGR", "OIL", "IND", "SER")
h = ("LAB", "CAP", "LAND", "NTR")
w = ("LAB", "LAND", "NTR")

def check_square():
    sam_small = sam.iloc[:, :-3]
    sam_small = sam_small.drop("TOTAL", errors="ignore")
    if sam_small.shape[0] != sam_small.shape[1]:
        raise ValueError(
            f"SAM is not square. It has {sam_small.shape[0]} rows and {sam_small.shape[1]} columns"
        )
    print("SAM is a square matrix.")

def row_col_equal():
    sam_small = sam.iloc[:, :-3]
    sam_small = sam_small.drop("TOTAL", errors="ignore")
    row_sum = sam_small.sum(axis=0)
    col_sum = sam_small.sum(axis=1)
    try:
        np.testing.assert_allclose(row_sum, col_sum)
        print("Row and column sums match.")
    except AssertionError as e:
        print("Row and column sums do not match.")
        print(e)

def runner(show_steps=True):
    # --- Model Solution Loop ---
    dist = 10
    tpi_iter = 0
    tpi_max_iter = 1000
    tpi_tol = 1e-10
    xi = 0.1

    pvec = np.ones(len(ind) + len(h))  # initial prices

    # --- Calibration ---
    d = calibrate.model_data(sam, h, ind)
    p = calibrate.parameters(d, ind)
    R = d.R0
    er = 1
    Zbar = d.Z0
    Ffbar = d.Ff0
    Kdbar = d.Kd0
    Qbar = d.Q0
    pdbar = pvec[0:len(ind)]
    pm = firms.eqpm(er, d.pWm)

    step_outputs = []
    while (dist > tpi_tol) and (tpi_iter < tpi_max_iter):
        tpi_iter += 1
        cge_args = [p, d, ind, h, Zbar, Qbar, Kdbar, pdbar, Ffbar, R, er]
        results = opt.root(
            cge.cge_system, pvec, args=cge_args, method="lm", tol=1e-5
        )
        pprime = results.x
        pyprime = pprime[0:len(ind)]
        pfprime = pprime[len(ind): len(ind) + len(h)]
        pyprime = Series(pyprime, index=list(ind))
        pfprime = Series(pfprime, index=list(h))

        pvec = pprime

        # Update variables
        pe = firms.eqpe(er, d.pWe)
        pm = firms.eqpm(er, d.pWm)
        pq = firms.eqpq(pm, pdbar, p.taum, p.eta, p.deltam, p.deltad, p.gamma)
        pz = firms.eqpz(p.ay, p.ax, pyprime, pq)
        Kk = agg.eqKk(pfprime, Ffbar, R, p.lam, pq)
        Td = gov.eqTd(p.taud, pfprime, Ffbar)
        Trf = gov.eqTrf(p.tautr, pfprime, Ffbar)
        Kf = agg.eqKf(Kk, Kdbar)
        Fsh = firms.eqFsh(R, Kf, er)
        Sp = agg.eqSp(p.ssp, pfprime, Ffbar, Fsh, Trf)
        I = hh.eqI(pfprime, Ffbar, Sp, Td, Fsh, Trf)
        E = firms.eqE(p.theta, p.xie, p.tauz, p.phi, pz, pe, Zbar)
        D = firms.eqDex(p.theta, p.xid, p.tauz, p.phi, pz, pdbar, Zbar)
        M = firms.eqM(p.gamma, p.deltam, p.eta, Qbar, pq, pm, p.taum)
        Qprime = firms.eqQ(p.gamma, p.deltam, p.deltad, p.eta, M, D)
        pdprime = firms.eqpd(p.gamma, p.deltam, p.eta, Qprime, pq, D)
        Zprime = firms.eqZ(p.theta, p.xie, p.xid, p.phi, E, D)
        Kdprime = agg.eqKd(d.g, Sp, p.lam, pq)
        Ffprime = d.Ff0.copy()
        Ffprime["CAP"] = R * Kk * (p.lam * pq).sum() / pfprime.loc["CAP"]

        dist = np.sqrt(((Zbar - Zprime) ** 2).sum())
        if show_steps:
            print(f"Iteration {tpi_iter} - Distance: {dist:.5e}")
            print(f"  Prices: {pyprime.to_dict()}")
            print(f"  Demand Q: {Qprime.to_dict()}")
        step_outputs.append((tpi_iter, dist, pyprime.copy(), Qprime.copy()))

        pdbar = xi * pdprime + (1 - xi) * pdbar
        Zbar = xi * Zprime + (1 - xi) * Zbar
        Kdbar = xi * Kdprime + (1 - xi) * Kdbar
        Qbar = xi * Qprime + (1 - xi) * Qbar
        Ffbar = xi * Ffprime + (1 - xi) * Ffbar

        Q = Qprime

        if dist < tpi_tol:
            print("Model converged.")
            break

    print("\nFinal Results:")
    print("Q (sectoral demand):")
    print(Q.to_markdown())
    print("Prices:")
    print(pyprime.to_markdown())
    return Q, pyprime, step_outputs

def project_future(Q_now, prices_now, years=10, growth_rate=0.02, price_inflation=0.01):
    # Simple projection: assume constant growth rate
    years_list = [f"Year_{i+1}" for i in range(years)]
    sectors = Q_now.index
    projections = pd.DataFrame(index=sectors, columns=years_list)
    price_proj = pd.DataFrame(index=sectors, columns=years_list)
    for s in sectors:
        demand, price = Q_now[s], prices_now[s]
        for y in range(years):
            demand = demand * (1 + growth_rate)
            price = price * (1 + price_inflation)
            projections.at[s, years_list[y]] = demand
            price_proj.at[s, years_list[y]] = price
    return projections, price_proj

if __name__ == "__main__":
    check_square()
    row_col_equal()
    Q, prices, steps = runner(show_steps=True)
    # Project 10-year demand and prices (simple constant rate)
    demand_proj, price_proj = project_future(Q, prices)
    print("\nDemand projection for next 10 years (2% annual growth):")
    print(demand_proj)
    print("\nPrice projection for next 10 years (1% annual inflation):")
    print(price_proj)
