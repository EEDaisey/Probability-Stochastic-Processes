# ##############################################################################
# Author:       Edward E. Daisey
# Project:      VerbVille.io – Extreme Value Theory (EVT) Analysis
# Description:  Statistical tail modeling of NASA HTTP request-per-second (RPS)
#               data using the Peaks-over-Threshold (POT) method and the 
#               Generalized Pareto Distribution (GPD).
#
# Objective:    1) Identify a high threshold u via the sample mean–excess plot.
#               2) Fit the GPD to exceedances above u via Maximum Likelihood.
#               3) Estimate return levels for capacity-planning purposes.
#               4) Generate visualizations for publication (both plots preserved).
#
# Dataset:      NASA HTTP Logs (July–Aug 1995) – Internet Traffic Archive
#               Preprocessed into 1-second RPS series in nasa_rps.csv.
# ##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genpareto

# ============================== Constants =====================================
CSV_FILE = "nasa_rps.csv"     # Preprocessed CSV file with 'timestamp' & 'rps'
THRESHOLD_PROB = 0.95         # 95th percentile threshold u95
RETURN_PERIOD_DAYS = 365      # "1 in 365 days" event
SECONDS_PER_DAY = 24 * 3600   # Observation resolution is 1 second
# =============================================================================


# ############################## Function 1 ###################################
# Function Name: LoadRpsData
# Function Purpose: Loads the NASA RPS dataset into memory as a NumPy array.
# Function Input:  csvFile - path to preprocessed nasa_rps.csv file.
# Function Output: rpsSeries (NumPy array), totalCount (int)
def LoadRpsData(csvFile: str):
    df = pd.read_csv(csvFile, parse_dates=["timestamp"])
    rpsSeries = df["rps"].to_numpy()
    return rpsSeries, rpsSeries.size
# ##############################################################################


# ############################## Function 2 ###################################
# Function Name: PlotMeanExcess
# Function Purpose: Computes & plots the sample mean–excess function e(u) for
#                   thresholds u in a given quantile range.
# Function Input:  rpsSeries - NumPy array of RPS values.
#                  probGrid  - list/array of quantile probabilities.
# Function Output: meanExcessPlot.png (displayed, optional save)
def PlotMeanExcess(rpsSeries: np.ndarray, probGrid: np.ndarray):
    us = np.quantile(rpsSeries, probGrid, method="higher")
    meanExcess = [(rpsSeries[rpsSeries > u] - u).mean() for u in us]

    plt.figure(figsize=(8, 4))
    plt.plot(us, meanExcess, "o-")
    plt.xlabel(r"Threshold $u$")
    plt.ylabel(r"Mean Excess $e(u)=\mathbb{E}[X-u\mid X>u]$")
    plt.title("Mean Excess Plot (Linearity ⇒ GPD)")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()
    return us
# ##############################################################################


# ############################## Function 3 ###################################
# Function Name: FitGpd
# Function Purpose: Fits a Generalized Pareto Distribution (GPD) to exceedances
#                   above a given threshold u using Maximum Likelihood Estimation.
# Function Input:  rpsSeries - NumPy array of RPS values.
#                  threshold - threshold u for exceedances.
# Function Output: xi (shape), sigma (scale), exceedances array, N_u (count), n_e
def FitGpd(rpsSeries: np.ndarray, threshold: float):
    exceedances = rpsSeries[rpsSeries > threshold] - threshold
    N_u = exceedances.size
    n_e = len(rpsSeries) / N_u  # mean spacing in seconds
    xi, loc, sigma = genpareto.fit(exceedances, floc=0)
    print(f"Estimated shape ξ={xi:.4f}, scale σ={sigma:.2f}")
    return xi, sigma, exceedances, N_u, n_e
# ##############################################################################


# ############################## Function 4 ###################################
# Function Name: ComputeReturnLevel
# Function Purpose: Computes the return level x_p for a given exceedance
#                   probability using the fitted GPD parameters.
# Function Input:  p        - target exceedance probability (per observation)
#                  u        - threshold
#                  xi       - GPD shape parameter
#                  sigma    - GPD scale parameter
#                  n_e      - mean spacing between exceedances
# Function Output: x_p      - estimated return level
#                  x_max    - finite upper endpoint if ξ < 0
def ComputeReturnLevel(p: float, u: float, xi: float, sigma: float, n_e: float):
    if abs(xi) < 1e-6:
        x_p = u + sigma * np.log(n_e / p)
    else:
        x_p = u + (sigma / xi) * ((n_e * p) ** (-xi) - 1)
    x_max = u - sigma / xi if xi < 0 else np.inf
    if x_p >= x_max:
        x_p = np.nextafter(x_max, -np.inf)
    return x_p, x_max
# ##############################################################################


# ############################## Function 5 ###################################
# Function Name: PlotGpdFit
# Function Purpose: Plots histogram of exceedances with the fitted GPD PDF overlay.
# Function Input:  exceedances - exceedances above threshold u
#                  xi, sigma   - fitted GPD parameters
# Function Output: Displayed plot (optional save)
def PlotGpdFit(exceedances: np.ndarray, xi: float, sigma: float):
    fig, ax = plt.subplots(figsize=(8, 4))
    counts, bins, _ = ax.hist(
        exceedances, bins=30, density=False, alpha=0.6, label="Empirical exceedances"
    )
    xVals = np.linspace(0, exceedances.max(), 200)
    pdfVals = genpareto.pdf(xVals, xi, loc=0, scale=sigma)
    binWidth = bins[1] - bins[0]
    ax.plot(
        xVals,
        pdfVals * len(exceedances) * binWidth,
        "r-",
        lw=2,
        label=f"GPD PDF (ξ={xi:.2f}, σ={sigma:.2f})",
    )
    ax.set_xlabel(r"Exceedance above threshold $u$")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of Exceedances & Fitted GPD (Counts)")
    ax.legend()
    plt.tight_layout()
    plt.show()
# ##############################################################################


# ############################## Function 6 ###################################
# Function Name: Main
# Function Purpose: Orchestrates the EVT analysis and generates all outputs.
# Function Input:  None (constants at top of script)
# Function Output: Printed stats, plots, return levels
def Main():
    # Step 1: Load Data
    rpsSeries, totalCount = LoadRpsData(CSV_FILE)

    # Step 2: Threshold Selection via Mean–Excess Plot
    probGrid = np.linspace(0.90, 0.99, 10)
    us = PlotMeanExcess(rpsSeries, probGrid)

    # Step 3: GPD Fit at u95
    u = np.quantile(rpsSeries, THRESHOLD_PROB, method="higher")
    xi, sigma, exceedances, N_u, n_e = FitGpd(rpsSeries, u)

    # Step 4: Return Level (per-day exceedance probability)
    p_day = 1 / SECONDS_PER_DAY
    p_year = 1 / (RETURN_PERIOD_DAYS * SECONDS_PER_DAY)
    x_p_day, x_max = ComputeReturnLevel(p_day, u, xi, sigma, n_e)
    x_p_year, _ = ComputeReturnLevel(p_year, u, xi, sigma, n_e)

    print(f"\n95th percentile threshold u = {u:.2f}")
    print(f"Exceedances N_u = {N_u}, spacing n_e = {n_e:.2f} seconds")
    print(f"Return level (1/day) ≈ {x_p_day:.2f} RPS")
    print(f"Return level (1/year) ≈ {x_p_year:.2f} RPS")
    print(f"Finite endpoint (if ξ<0): x_max = {x_max:.2f}")

    # Step 5: Plot GPD Fit
    PlotGpdFit(exceedances, xi, sigma)

    # Step 6: Five-number summary
    summary = pd.Series(
        np.percentile(rpsSeries, [0, 25, 50, 75, 100]),
        index=["min", "Q1", "median", "Q3", "max"],
    )
    print("\nFive-Number Summary:")
    print(summary)
# ##############################################################################


# ============================ Run Analysis ===================================
if __name__ == "__main__":
    Main()

#  ##### Results #####
#  Estimated shape ξ=-0.1131, scale σ=1.82

#  95th percentile threshold u = 4.00
#  Exceedances N_u = 55522, spacing n_e = 37.38 seconds
#  Return level (1/day) ≈ 13.39 RPS
#  Return level (1/year) ≈ 16.65 RPS
#  Finite endpoint (if ξ<0): x_max = 20.09

#  Five-Number Summary:
#  min        1.0
#  Q1         1.0
#  median     1.0
#  Q3         2.0
#  max       20.0
#  dtype: float64    
# =============================================================================
