import scipy.optimize as opt
import numpy as np
import pandas as pd
from pandas import Series
import os

# Make sure you have open_cge installed in your environment or in the same directory
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
    "AGR", "OIL", "IND", "SER", "LAB", "CAP", "LAND", "NTR", "DTX", "IDT", "ACT",
    "HOH", "GOV", "INV", "EXT",
)
ind = ("AGR", "OIL", "IND", "SER")
h = ("LAB", "CAP", "LAND", "NTR")
w = ("LAB", "LAND", "NTR")

def check_square():
    sam_small = sam.iloc[:, :-3]
    sam_small = sam_small.drop("TOTAL")
    if not sam_small.shape[0] == sam_small.shape[1]:
        raise ValueError(
            f"SAM is not square. It has {sam_small.shape[0]} rows and {sam_small.shape[1]} columns"
        )

def row_col_equal():
    sam_small = sam.iloc[:, :-3]
    sam_small = sam_small.drop("TOTAL")
    row_sum = sam_small.sum(axis=0)
    col_sum = sam_small.sum(axis=1)
    np.testing.assert_allclose(row_sum, col_sum)

def runner(verbose=True):
    """
    Solves the CGE model and projects demand/price for 10 years
    """

    tpi_tol = 1e-10
    tpi_max_iter = 1000
    xi = 0.1

    # For projection
    n_years = 10
    growth_rate = 0.03  # Example: 3% annual growth in exogenous variables

    # Initial values
    pvec = np.ones(len(ind) + len(h))
    d = calibrate.model_data(sam, h, ind)
    p = calibrate.parameters(d, ind)

    # Collect results for each year
    results_dict = {
        "Year": [],
        "Demand": [],
        "Prices": [],
    }

    # Initial conditions
    Zbar = d.Z0
    Ffbar = d.Ff0
    Kdbar = d.Kd0
    Qbar = d.Q0
    pdbar = pvec[0 : len(ind)]

    # Simulation over 10 years
    for year in range(1, n_years + 1):
        dist = 10
        tpi_iter = 0
        er = 1
        R = d.R0
        pm = firms.eqpm(er, d.pWm)

        while (dist > tpi_tol) & (tpi_iter < tpi_max_iter):
            tpi_iter += 1
            cge_args = [p, d, ind, h, Zbar, Qbar, Kdbar, pdbar, Ffbar, R, er]
            if verbose and tpi_iter == 1:
                print(f"\n=== Year {year} Iteration {tpi_iter} ===")
                print("Initial price vector: ", pvec)
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
            Ffprime = d.Ff0
            Ffprime["CAP"] = R * Kk * (p.lam * pq).sum() / pfprime.iloc[1]

            dist = (((Zbar - Zprime) ** 2) ** (1 / 2)).sum()
            if verbose:
                print(f"Iteration {tpi_iter}: Distance = {dist}")

            # Dampening for stability
            pdbar = xi * pdprime + (1 - xi) * pdbar
            Zbar = xi * Zprime + (1 - xi) * Zbar
            Kdbar = xi * Kdprime + (1 - xi) * Kdbar
            Qbar = xi * Qprime + (1 - xi) * Qbar
            Ffbar = xi * Ffprime + (1 - xi) * Ffbar

            if dist < tpi_tol:
                if verbose:
                    print(f"Model converged at iteration {tpi_iter}")

        # Store results for this year
        results_dict["Year"].append(year)
        results_dict["Demand"].append(Qbar.values.copy())
        results_dict["Prices"].append(pdbar.copy())

        # Show year results
        print(f"\nYear {year} Results:")
        print("Demand by sector:", dict(zip(ind, Qbar)))
        print("Prices by sector:", dict(zip(ind, pdbar)))

        # Project exogenous growth in demand as example
        Qbar = Qbar * (1 + growth_rate)
        # Optionally: Adjust prices/exogenous variables as needed

    # Convert results to DataFrames for plotting/saving
    demand_df = pd.DataFrame(
        np.vstack(results_dict["Demand"]), columns=ind, index=results_dict["Year"]
    )
    price_df = pd.DataFrame(
        np.vstack(results_dict["Prices"]), columns=ind, index=results_dict["Year"]
    )

    print("\n== Final 10-year Demand Projection ==")
    print(demand_df)
    print("\n== Final 10-year Price Projection ==")
    print(price_df)
    return demand_df, price_df

if __name__ == "__main__":
    check_square()
    row_col_equal()
    demand, price = runner(verbose=True)
