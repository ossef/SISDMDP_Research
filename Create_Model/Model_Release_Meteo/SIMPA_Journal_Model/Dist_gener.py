#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Global script: from PVWatts hourly CSV -> weather regimes (M=0..3),
- builds discrete arrival distributions P(A=a | M, h) WITHOUT smoothing,
- estimates a 4x4 daily weather-regime transition matrix P(M_{t+1} | M_t),
- writes:
    * one .data file per regime (active hours only),
      BUT with a FIXED arrivals support across all regimes:
        - support size is the maximum Amax observed across all regimes (over their active hours)
        - regimes with smaller Amax are padded with zeros (for a > Amax_regime)
    * one Day_Change.data file (4x4),
- generates one PDF per regime with:
    * first page: mean number of packets vs hour (0..23)
    * next pages: barplots (6 plots per page), one plot per active hour.
  (PDFs keep regime-specific Amax for readability, i.e., unchanged.)

Inputs:
- PVWatts hourly CSV with metadata header.
- Uses column: 'AC System Output (W)'

Outputs (example city="Barcelona"):
- ./NREL_Extracts/<CITY>/<CITY>_M0.data ... <CITY>_M3.data   (fixed support size)
- ./NREL_Extracts/<CITY>/<CITY>_M0_barplots.pdf ...         (unchanged plotting scale per regime)
- ./NREL_Extracts/<CITY>/<CITY>_Day_Change.data
- ./NREL_Extracts/Service_Demand.data and Service_Demand.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ---------------------------
# User parameters
# ---------------------------

#CITY = "Santiago"  # Paris, Rabat, Barcelona, Moscow, Unalaska, London, Texas, Athens, Reykjavik, Yaounde, Santiago
#OUT_DIR = "./NREL_Extracts/" + CITY
#CSV_PATH = "./NREL_Data/" + CITY + "_pvwatts_hourly.csv"

COL_W = "AC System Output (W)"
PACKET_WH = 800                 # packet size (Wh)
PROB_ENERGY_THRESHOLD = 0       # hour is active if raw P(A>0) > threshold

# If you want a fixed window regardless of threshold, set:
FORCE_HOURS = True
FORCE_START_H = 6
FORCE_END_H = 20

# Plot layout: 6 figures per page (2 columns x 3 rows)
PLOTS_PER_PAGE_COLS = 2
PLOTS_PER_PAGE_ROWS = 3


# ---------------------------
# Helpers
# ---------------------------

REGIME_NAMES = {
    0: "Very cloudy",
    1: "Cloudy",
    2: "Partly cloudy",
    3: "Clear sky",
}


def read_pvwatts_hourly(csv_path: str) -> pd.DataFrame:
    """
    PVWatts CSV includes metadata lines; header starts at line ~32 (index 31).
    """
    df = pd.read_csv(csv_path, skiprows=31)
    required = {"Month", "Day", "Hour", COL_W}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found columns: {list(df.columns)}")

    # Build synthetic timestamp for sorting and day grouping (dummy year: typical meteorological year)
    df["ts"] = pd.to_datetime(
        dict(
            year=2025,  # dummy year; PVWatts provides a typical year, not a specific calendar year
            month=df["Month"],
            day=df["Day"],
            hour=df["Hour"],
        )
    )
    df["date"] = df["ts"].dt.date
    df["month"] = df["ts"].dt.month
    df["hour"] = df["ts"].dt.hour

    # W -> kWh per hour (slot=1h)
    df["kwh"] = df[COL_W] / 1000.0
    return df


def label_regimes_by_normalized_daily_energy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Labels each day into M in {0,1,2,3} using Enorm = Eday / P95(Eday | month)
    then global quantiles (25/50/75%) of Enorm.
    """
    Eday = (
        df.groupby(["month", "date"], as_index=False)["kwh"]
        .sum()
        .rename(columns={"kwh": "Eday"})
    )

    p95 = Eday.groupby("month")["Eday"].quantile(0.95).rename("p95").reset_index()
    Eday = Eday.merge(p95, on="month", how="left")
    Eday["Enorm"] = Eday["Eday"] / Eday["p95"].replace(0, np.nan)

    q = Eday["Enorm"].quantile([0.25, 0.50, 0.75]).values

    def lab(en):
        if en < q[0]:
            return 0
        if en < q[1]:
            return 1
        if en < q[2]:
            return 2
        return 3

    Eday["M"] = Eday["Enorm"].apply(lab)
    df2 = df.merge(Eday[["date", "M"]], on="date", how="left")
    return df2


def estimate_weather_transition_matrix(df_labeled: pd.DataFrame) -> np.ndarray:
    """
    Estimates a 4x4 Markov transition matrix P:
    P[i,j] = P(M_{t+1}=j | M_t=i)
    """
    daily = (
        df_labeled.groupby("date", as_index=False)["M"]
        .first()
        .sort_values("date")
        .reset_index(drop=True)
    )

    M_seq = daily["M"].to_numpy(dtype=int)
    counts = np.zeros((4, 4), dtype=float)

    for t in range(len(M_seq) - 1):
        i = int(M_seq[t])
        j = int(M_seq[t + 1])
        counts[i, j] += 1.0

    P = np.zeros((4, 4), dtype=float)
    for i in range(4):
        s = counts[i, :].sum()
        if s > 0:
            P[i, :] = counts[i, :] / s
        else:
            P[i, i] = 1.0

    return P


def write_day_change_file(city: str, P: np.ndarray, out_dir: str) -> str:
    """
    Writes the 4x4 transition matrix to: <city>_Day_Change.data
    """
    path = os.path.join(out_dir, f"{city}_Day_Change.data")
    with open(path, "w", encoding="utf-8") as f:
        f.write("4\n")
        for i in range(4):
            f.write(" ".join(f"{P[i, j]:.10f}" for j in range(4)) + "\n")
    return path


def build_pmf_and_raw_activity(df: pd.DataFrame, packet_wh: int):
    """
    Builds:
    - A(t) in packets (rounded)
    - pmf[m,h,a] with NO smoothing (pure empirical frequencies)
    - raw_p_pos[m,h] = raw frequency P(A>0 | m,h)

    Note: If a (m,h) has no samples, pmf[m,h,0]=1 by convention.
    """
    packet_kwh = packet_wh / 1000.0

    df = df.copy()
    df["A"] = np.rint(df["kwh"] / packet_kwh).astype(int)

    Amax_global = int(df["A"].max())
    pmf = np.zeros((4, 24, Amax_global + 1), dtype=float)
    raw_p_pos = np.zeros((4, 24), dtype=float)

    for m in range(4):
        for h in range(24):
            x = df[(df["M"] == m) & (df["hour"] == h)]["A"].values
            if len(x) > 0:
                raw_p_pos[m, h] = float(np.mean(x > 0))
                counts = np.bincount(x, minlength=Amax_global + 1).astype(float)
                s = counts.sum()
                if s > 0:
                    pmf[m, h, :] = counts / s
                else:
                    pmf[m, h, 0] = 1.0
            else:
                raw_p_pos[m, h] = 0.0
                pmf[m, h, 0] = 1.0

    return df, pmf, raw_p_pos


def select_hours_for_regime(m: int, raw_p_pos: np.ndarray) -> list[int]:
    """
    Returns list of hours to include (active hours).
    """
    if FORCE_HOURS:
        return list(range(FORCE_START_H, FORCE_END_H + 1))
    return [h for h in range(24) if raw_p_pos[m, h] > PROB_ENERGY_THRESHOLD]


def compute_Amax_for_regime(df: pd.DataFrame, m: int, hours: list[int]) -> int:
    """
    Max packet count observed in raw data (for this regime + selected hours)
    """
    subset = df[(df["M"] == m) & (df["hour"].isin(hours))]
    if len(subset) == 0:
        return 0
    return int(subset["A"].max())


def write_data_file_fixed_support(
    city: str,
    m: int,
    hours: list[int],
    pmf: np.ndarray,
    nb_paquets_fixed: int,
    packet_wh: int,
    out_dir: str,
) -> str:
    """
    Writes: City_Mm.data (active hours only)
    BUT with a fixed support size across regimes: a=0..nb_paquets_fixed-1.

    For a beyond what is actually present in regime/hour, pmf entries are 0 already
    (since pmf is built on a global Amax). If nb_paquets_fixed exceeds pmf depth,
    values are padded with zeros.
    """
    h_start, h_end = hours[0], hours[-1]
    fname = f"{city}_M{m}.data"
    path = os.path.join(out_dir, fname)

    with open(path, "w", encoding="utf-8") as f:
        f.write("Matrice des probabilites (heure x paquets) :\n")
        f.write(f"{h_start} {h_end} {nb_paquets_fixed} {packet_wh}\n")
        f.write("Heure\t" + "\t".join(str(a) for a in range(nb_paquets_fixed)) + "\n")

        for h in range(h_start, h_end + 1):
            probs = pmf[m, h, :]
            # Write exactly nb_paquets_fixed probabilities, pad with 0 if needed
            row = []
            for a in range(nb_paquets_fixed):
                p = probs[a] if a < probs.shape[0] else 0.0
                row.append(f"{p:.20f}")
            f.write(str(h) + "\t" + "\t".join(row) + "\n")

    return path


def _plot_hour_bar(ax, m: int, h: int, pmf: np.ndarray, Amax_regime: int):
    probs = pmf[m, h, :Amax_regime + 1]
    a = np.arange(Amax_regime + 1)
    mean_packets = float((a * probs).sum())

    ax.bar(a, probs)
    ax.set_xlabel("Number of energy packets")
    ax.set_ylabel("Probability")
    ax.set_title(f"Hour {h:02d}:00 - Average {mean_packets:.2f}")

    step = max(1, (Amax_regime + 1) // 8)
    ax.set_xticks(np.arange(0, Amax_regime + 1, step))


def _plot_mean_packets_by_hour(
    ax,
    city: str,
    m: int,
    hours_active: list[int],
    pmf: np.ndarray,
    Amax_regime: int,
    packet_wh: int,
):
    """
    First-page summary plot (0..23) (unchanged).
    """
    all_hours = np.arange(24)
    active_set = set(hours_active)

    means_all = np.zeros(24, dtype=float)
    for h in all_hours:
        if h in active_set:
            probs = pmf[m, h, :Amax_regime + 1]
            a = np.arange(Amax_regime + 1)
            means_all[h] = float((a * probs).sum())
        else:
            means_all[h] = 0.0

    if len(hours_active) > 0:
        active_vals = means_all[hours_active]
        vmin = float(active_vals.min())
        vmax = float(active_vals.max())
    else:
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-12

    norm = plt.Normalize(vmin, vmax)
    base_color = plt.cm.inferno(0.6)

    min_alpha = 0.20
    max_alpha = 1.0

    for h in all_hours:
        value = means_all[h]
        if value <= 0:
            alpha = min_alpha
        else:
            intensity = norm(value)
            alpha = min_alpha + intensity * (max_alpha - min_alpha)
        ax.bar(h, value, color=base_color, alpha=alpha)  # no edgecolor

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean number of energy packets")
    ax.set_xticks(np.arange(0, 24, 1))
    ax.set_xlim(-0.5, 23.5)

    ax.set_title(
        f"{city} – Regime M{m} ({REGIME_NAMES[m]})\n"
        f"Average energy packets per hour (packet = {packet_wh} Wh)",
        fontsize=14,
    )
    ax.grid(axis="y", alpha=0.3)


def make_pdf_barplots(
    city: str,
    m: int,
    hours: list[int],
    Amax_regime: int,
    pmf: np.ndarray,
    packet_wh: int,
    out_dir: str,
) -> str:
    """
    Generates PDF per regime (unchanged plotting scale per regime).
    """
    pdf_path = os.path.join(out_dir, f"{city}_M{m}_barplots.pdf")

    cols = PLOTS_PER_PAGE_COLS
    rows = PLOTS_PER_PAGE_ROWS
    per_page = cols * rows

    with PdfPages(pdf_path) as pdf:
        fig0, ax0 = plt.subplots(figsize=(10, 6))
        _plot_mean_packets_by_hour(ax0, city, m, hours, pmf, Amax_regime, packet_wh)
        fig0.tight_layout()
        pdf.savefig(fig0, bbox_inches="tight")
        plt.close(fig0)

        for i in range(0, len(hours), per_page):
            chunk = hours[i:i + per_page]

            fig, axes = plt.subplots(rows, cols, figsize=(11.69, 8.27))  # ~A4 landscape
            axes = np.array(axes).reshape(-1)

            for k in range(per_page):
                ax = axes[k]
                if k < len(chunk):
                    h = chunk[k]
                    _plot_hour_bar(ax, m, h, pmf, Amax_regime)
                else:
                    ax.axis("off")

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return pdf_path


def Service_Demand():
    """
    Writes:
      ./NREL_Extracts/Service_Demand.data
      ./NREL_Extracts/Service_Demand.pdf
    """
    data = {
        "Hour": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "Total_Erlangs": [30, 40, 45, 54.6, 60.6, 58.3, 45.2, 51.0, 59.2, 55.7, 52.7, 45, 40, 35, 30],
    }
    df = pd.DataFrame(data)

    total_erlangs = df["Total_Erlangs"].sum()
    df["P_relative_arrival"] = df["Total_Erlangs"] / total_erlangs

    first_hour = df["Hour"].min()
    last_hour = df["Hour"].max()

    os.makedirs("./NREL_Extracts", exist_ok=True)

    with open("./NREL_Extracts/Service_Demand.data", "w") as file:
        file.write(f"{first_hour} {last_hour}\n")
        for _, row in df.iterrows():
            file.write(f"{int(row['Hour'])} {row['P_relative_arrival']:.5f}\n")

    norm = plt.Normalize(df["P_relative_arrival"].min(), df["P_relative_arrival"].max())
    base_color = plt.cm.inferno(0.6)

    min_alpha = 0.2
    max_alpha = 1.0

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, value in enumerate(df["P_relative_arrival"]):
        intensity = norm(value)
        alpha = min_alpha + (intensity * (max_alpha - min_alpha))
        ax.bar(df["Hour"][i], value, color=base_color, alpha=alpha)

    plt.xlabel("Hour")
    plt.ylabel("Service demand probability")
    plt.grid(True)
    plt.xticks(df["Hour"])
    plt.savefig("./NREL_Extracts/Service_Demand.pdf")
    plt.close(fig)


# ---------------------------
# Main
# ---------------------------

def main():
    cities = ["Paris", "Rabat", "Barcelona", "Moscow", "Unalaska", "London",
              "Tokyo", "Athens", "Reykjavik", "Yaounde", "Santiago"]

    created_global = []

    # --- Service Demand hourly (unique, once) ---
    Service_Demand()
    created_global.append("./NREL_Extracts/Service_Demand.data")
    created_global.append("./NREL_Extracts/Service_Demand.pdf")

    for CITY in cities:
        print(f"\n==============================")
        print(f"Processing CITY = {CITY}")
        print(f"==============================")

        OUT_DIR = "./NREL_Extracts/" + CITY
        CSV_PATH = "./NREL_Data/" + CITY + "_pvwatts_hourly.csv"

        os.makedirs(OUT_DIR, exist_ok=True)

        # --- Read + label regimes ---
        df = read_pvwatts_hourly(CSV_PATH)
        df = label_regimes_by_normalized_daily_energy(df)

        # --- Weather regime day-to-day transition matrix ---
        P_M = estimate_weather_transition_matrix(df)
        day_change_path = write_day_change_file(CITY, P_M, OUT_DIR)

        df, pmf, raw_p_pos = build_pmf_and_raw_activity(df, PACKET_WH)

        created = [day_change_path]

        # --- Determine a FIXED arrivals support across regimes (max Amax over active hours) ---
        hours_by_regime = {}
        Amax_by_regime = {}
        for m in range(4):
            hours = select_hours_for_regime(m, raw_p_pos)
            hours_by_regime[m] = hours
            if hours:
                Amax_by_regime[m] = compute_Amax_for_regime(df, m, hours)
            else:
                Amax_by_regime[m] = 0

        Amax_fixed = max(Amax_by_regime.values())
        nb_paquets_fixed = Amax_fixed + 1

        # --- Write .data and PDFs for each regime ---
        for m in range(4):
            hours = hours_by_regime[m]
            if not hours:
                print(f"[WARN] No active hours for M{m}. Skipping.")
                continue

            Amax_regime = Amax_by_regime[m]

            # .data uses fixed support size across regimes (pad with zeros)
            data_path = write_data_file_fixed_support(
                CITY, m, hours, pmf, nb_paquets_fixed, PACKET_WH, OUT_DIR
            )
            created.append(data_path)

            # PDFs remain regime-specific (unchanged)
            pdf_path = make_pdf_barplots(CITY, m, hours, Amax_regime, pmf, PACKET_WH, OUT_DIR)
            created.append(pdf_path)

            print(
                f"[OK] {CITY} M{m}: wrote {os.path.basename(data_path)} (support 0..{Amax_fixed}) "
                f"and {os.path.basename(pdf_path)} (support 0..{Amax_regime})"
            )

        print(f"[OK] {CITY}: wrote {os.path.basename(day_change_path)}")

        print(f"\nCreated files for {CITY}:")
        for p in created:
            print(" -", p)

        created_global.extend(created)

    print("\n==============================")
    print("[DONE] Created files (global list):")
    for p in created_global:
        print(" -", p)


if __name__ == "__main__":
    main()
