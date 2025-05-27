import scipy.optimize as opt
import numpy as np
import pandas as pd
from pandas import Series
import os

# Make sure open_cge is installed and available in your environment
from open_cge import government as gov
from open_cge import household as hh
from open_cge import aggregates as agg
from open_cge import firms, calibrate
from open_cge import simpleCGE as cge

# load social accounting matrix
current_path = os.path.abspath(os.path.dirname(__file__))
sam_path = os.path.join(current_path, "SAM.xlsx")
sam = pd.read_excel(sam_path, index_col=0, header=0)

# declare sets of variables
u = (
    "AGR", "OIL", "IND", "SER", "LAB", "CAP", "LAND", "NTR",
    "DTX", "IDT", "ACT", "HOH", "GOV", "INV", "EXT"
)
ind = ("AGR", "OIL", "IND", "SER")
h = ("LAB", "CAP", "LAND", "NTR")
w = ("LAB", "LAND", "NTR")

def check_square():
    sam_small = sam.iloc[:, :-3]
    sam_small = sam_small.drop("TOTAL", errors='ignore')
    if not sam_small.shape[0] == sam_small.shape[1]:
        raise ValueError(
            f"SAM is not square. It has {sam_small.shape[0]} rows and {sam_small.shape[1]} columns"
        )
    print("Check Square: PASSED")

def row_total():
    sam_small = sam.iloc[:, :-3]
    sam_small = sam_small.drop("TOTAL", errors='ignore')
    row_sum = sam_small.sum(axis=0)
    print("Row totals:\n", row_sum)
    return row_sum

def col_total():
    sam_small = sam.iloc[:, :-3]
    sam_small = sam_small.drop("TOTAL", errors='ignore')
    col_sum = sam_small.sum(axis=1)
    print("Column totals:\n", col_sum)
    return col_sum

def row_col_equal():
    sam_small = sam.iloc[:, :-3]
    sam_small = sam_small.drop("TOTAL", errors='ignore')
    row_sum = sam_small.sum(axis=0)
    col_sum = sam_small.sum(axis=1)
    np.testing.assert_allclose(row_sum, col_sum)
    print("Row/Col totals are equal: PASSED")

def runner():
    dist = 10
    tpi_iter = 0
    tpi_max_iter = 1000
    tpi_tol = 1e-10
    xi = 0.1
    pvec = np.ones(len(ind) + len(h))

    d = calibrate.model_data(sam, h, ind)
    p = calibrate.parameters(d, ind)

    R = d.R0
    er = 1

    Zbar = d.Z0
    Ffbar = d.Ff0
    Kdbar = d.Kd0
    Qbar = d.Q0
    pdbar = pvec[0 : len(ind)]

    pm = firms.eqpm(er, d.pWm)

    print("\n--- Solving CGE system ---")
    while (dist > tpi_tol) & (tpi_iter < tpi_max_iter):
        tpi_iter += 1
        cge_args = [p, d, ind, h, Zbar, Qbar, Kdbar, pdbar, Ffbar, R, er]

        print(f"\nIteration {tpi_iter}")
        print("Initial guess:", pvec)
        results = opt.root(
            cge.cge_system, pvec, args=cge_args, method="lm", tol=1e-5
        )
        pprime = results.x
        pyprime = pprime[0 : len(ind)]
        pfprime = pprime[len(ind) : len(ind) + len(h)]
        pyprime = Series(pyprime, index=list(ind))
        pfprime = Series(pfprime, index=list(h))

        pvec = pprime

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
        # Avoid division by zero in pfprime
        if pfprime.iloc[1] != 0:
            Ffprime["CAP"] = R * Kk * (p.lam * pq).sum() / pfprime.iloc[1]

        dist = (((Zbar - Zprime) ** 2) ** (1 / 2)).sum()
        print(f"Distance: {dist:.6g}")
        print(f"  Qprime: \n{Qprime}")

        pdbar = xi * pdprime + (1 - xi) * pdbar
        Zbar = xi * Zprime + (1 - xi) * Zbar
        Kdbar = xi * Kdprime + (1 - xi) * Kdbar
        Qbar = xi * Qprime + (1 - xi) * Qbar
        Ffbar = xi * Ffprime + (1 - xi) * Ffbar

        Q = firms.eqQ(p.gamma, p.deltam, p.deltad, p.eta, M, D)

        # Optionally, break if distance is very small to avoid unnecessary iterations
        if dist < 1e-5:
            break

    print("\nModel solved! Final Q (sector outputs):\n", Q)
    return Q

def demand_projection(Q, growth_rate=0.03, years=5):
    """
    Simple projection for demand (output) for next 5 years with assumed growth rate.
    """
    projections = pd.DataFrame(index=range(1, years+1), columns=Q.index)
    for year in range(1, years+1):
        projections.loc[year] = Q * ((1 + growth_rate) ** year)
    print("\n--- Demand Projections for Each Sector (Next 5 Years, 3% Annual Growth) ---")
    print(projections)
    return projections

if __name__ == "__main__":
    check_square()
    row_col_equal()
    Q = runner()
    projections = demand_projection(Q)
