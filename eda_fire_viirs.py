"""
Exploratory Data Analysis - VIIRS Active Fire Archive 2018
Dataset: fire_viirs_2018.parquet  (SNPP VIIRS Collection 2, SV-C2)
         19,170,193 fire detections, global, 2018-01-01 to 2018-12-31

Outputs saved to: eda_output/
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Force stdout to UTF-8 on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
OUT_DIR = "eda_output"
os.makedirs(OUT_DIR, exist_ok=True)

PARQUET = "data/processed/fire_viirs_2018.parquet"
SEED    = 42
rng     = np.random.default_rng(SEED)

TYPE_MAP = {0: "Presumed vegetation", 1: "Active volcano",
            2: "Other static land",   3: "Offshore"}

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

print("Loading parquet ...")
df = pd.read_parquet(PARQUET)

df["ACQ_DATE"] = pd.to_datetime(df["ACQ_DATE"])
df["MONTH"]    = df["ACQ_DATE"].dt.month
df["DOY"]      = df["ACQ_DATE"].dt.dayofyear
df["WEEK"]     = df["ACQ_DATE"].dt.isocalendar().week.astype(int)
df["HEMI"]     = np.where(df["LATITUDE"] >= 0, "Northern", "Southern")
df["LOG_FRP"]  = np.log1p(df["FRP"])

print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")


# ---------------------------------------------------------------------------
# 2. Dataset overview
# ---------------------------------------------------------------------------

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Total detections : {len(df):>15,}")
print(f"Date range       : {df['ACQ_DATE'].min().date()} to {df['ACQ_DATE'].max().date()}")
print(f"Latitude range   : {df['LATITUDE'].min():.2f} to {df['LATITUDE'].max():.2f}")
print(f"Longitude range  : {df['LONGITUDE'].min():.2f} to {df['LONGITUDE'].max():.2f}")
print(f"Missing values   : {df.isnull().sum().sum()}")
print()

stats = df[["LATITUDE","LONGITUDE","BRIGHTNESS","BRIGHT_T31","FRP","SCAN","TRACK"]].describe().round(3)
print(stats.to_string())
stats.to_csv(os.path.join(OUT_DIR, "01_descriptive_stats.csv"))

cat_cols = ["CONFIDENCE", "DAYNIGHT", "TYPE", "MONTH"]
for c in cat_cols:
    vc = df[c].value_counts(normalize=True) * 100
    vc.name = "pct"
    print(f"\n--- {c} ---")
    print(vc.round(2).to_string())
    vc.reset_index().to_csv(os.path.join(OUT_DIR, f"01_{c.lower()}_dist.csv"), index=False)

print()


# ---------------------------------------------------------------------------
# 3. Figure 1 - Monthly fire counts (total & by hemisphere)
# ---------------------------------------------------------------------------

monthly = (df.groupby(["MONTH","HEMI"])
             .size()
             .reset_index(name="count"))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

totals = monthly.groupby("MONTH")["count"].sum()
axes[0].bar(totals.index, totals.values / 1e6, color="#E07B54", edgecolor="white")
axes[0].set_xticks(range(1, 13))
axes[0].set_xticklabels(MONTH_NAMES, rotation=30, ha="right")
axes[0].set_xlabel("Month")
axes[0].set_ylabel("Fire detections (millions)")
axes[0].set_title("Monthly Fire Detections - Global 2018")
for i, v in enumerate(totals.values):
    axes[0].text(i+1, v/1e6 + 0.02, f"{v/1e6:.2f}M", ha="center", fontsize=8)

pivot = monthly.pivot(index="MONTH", columns="HEMI", values="count").fillna(0)
x = np.arange(1, 13)
w = 0.4
axes[1].bar(x - w/2, pivot.get("Northern", 0)/1e6, w, label="Northern", color="#4E9AF1")
axes[1].bar(x + w/2, pivot.get("Southern", 0)/1e6, w, label="Southern", color="#F1744E")
axes[1].set_xticks(x)
axes[1].set_xticklabels(MONTH_NAMES, rotation=30, ha="right")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Fire detections (millions)")
axes[1].set_title("Monthly Detections by Hemisphere")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_monthly_fire_counts.png"), dpi=150)
plt.close()
print("Saved: 02_monthly_fire_counts.png")


# ---------------------------------------------------------------------------
# 4. Figure 2 - Spatial distribution (sampled scatter)
# ---------------------------------------------------------------------------

N_SAMPLE = 200_000
idx   = rng.choice(len(df), N_SAMPLE, replace=False)
samp  = df.iloc[idx].copy()

fig, ax = plt.subplots(figsize=(16, 8))
sc = ax.scatter(
    samp["LONGITUDE"], samp["LATITUDE"],
    c=samp["LOG_FRP"], cmap="YlOrRd",
    s=0.5, alpha=0.4, linewidths=0,
)
cb = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
cb.set_label("log(1 + FRP)  [MW]")
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(
    f"Global Fire Detections - VIIRS 2018  (sample: {N_SAMPLE:,} / {len(df):,})",
    fontsize=12,
)
ax.axhline(0, color="grey", lw=0.6, ls="--")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_spatial_distribution.png"), dpi=150)
plt.close()
print("Saved: 03_spatial_distribution.png")


# ---------------------------------------------------------------------------
# 5. Figure 3 - FRP distribution
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].hist(df["FRP"].clip(0, 200), bins=100, color="#E07B54", edgecolor="white", linewidth=0.3)
axes[0].set_xlabel("FRP (MW)  [clipped at 200]")
axes[0].set_ylabel("Count")
axes[0].set_title("FRP Distribution (linear scale)")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

axes[1].hist(df["LOG_FRP"], bins=100, color="#4E9AF1", edgecolor="white", linewidth=0.3)
axes[1].set_xlabel("log(1 + FRP)")
axes[1].set_ylabel("Count")
axes[1].set_title("log(1+FRP) Distribution")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

conf_vals = df["CONFIDENCE"].unique().tolist()
conf_order = [c for c in ["low","nominal","high"] if c in conf_vals]
sns.boxplot(
    data=df[df["FRP"] < 100],
    x="CONFIDENCE", y="FRP",
    order=conf_order,
    palette="Set2", ax=axes[2],
)
axes[2].set_xlabel("Confidence")
axes[2].set_ylabel("FRP (MW)  [FRP < 100]")
axes[2].set_title("FRP by Detection Confidence")

plt.suptitle("Fire Radiative Power (FRP) - VIIRS 2018", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_frp_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 04_frp_distribution.png")


# ---------------------------------------------------------------------------
# 6. Figure 4 - Brightness temperature
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].hist(df["BRIGHTNESS"], bins=80, color="#E07B54", edgecolor="white", linewidth=0.3)
axes[0].set_xlabel("Brightness T21 (K)")
axes[0].set_ylabel("Count")
axes[0].set_title("T21 Brightness Temperature")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

axes[1].hist(df["BRIGHT_T31"], bins=80, color="#4E9AF1", edgecolor="white", linewidth=0.3)
axes[1].set_xlabel("Brightness T31 (K)")
axes[1].set_ylabel("Count")
axes[1].set_title("T31 Brightness Temperature")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

samp2 = df.sample(50_000, random_state=SEED)
axes[2].scatter(
    samp2["BRIGHT_T31"], samp2["BRIGHTNESS"],
    s=0.5, alpha=0.2, color="#555",
)
axes[2].set_xlabel("T31 (K)")
axes[2].set_ylabel("T21 (K)")
axes[2].set_title("T21 vs T31 (50K sample)")
corr = df["BRIGHTNESS"].corr(df["BRIGHT_T31"])
axes[2].text(0.05, 0.95, f"r = {corr:.3f}", transform=axes[2].transAxes,
             fontsize=10, va="top")

plt.suptitle("Brightness Temperatures - VIIRS 2018", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_brightness_temp.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 05_brightness_temp.png")


# ---------------------------------------------------------------------------
# 7. Figure 5 - Weekly time series by hemisphere
# ---------------------------------------------------------------------------

weekly = (df.groupby(["WEEK","HEMI"])
            .agg(count=("FRP","count"), frp_mean=("FRP","mean"))
            .reset_index())

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for hemi, color in [("Northern","#4E9AF1"), ("Southern","#F1744E")]:
    sub = weekly[weekly["HEMI"] == hemi]
    axes[0].plot(sub["WEEK"], sub["count"]/1e3, label=hemi, color=color, lw=1.5)
    axes[1].plot(sub["WEEK"], sub["frp_mean"], label=hemi, color=color, lw=1.5)

axes[0].set_ylabel("Detections (thousands)")
axes[0].set_title("Weekly Fire Detections by Hemisphere")
axes[0].legend()

axes[1].set_ylabel("Mean FRP (MW)")
axes[1].set_title("Weekly Mean Fire Radiative Power by Hemisphere")
axes[1].legend()
axes[1].set_xlabel("Week of year")

week_starts = [1, 5, 9, 14, 18, 22, 26, 31, 35, 39, 44, 48]
axes[1].set_xticks(week_starts)
axes[1].set_xticklabels(MONTH_NAMES, rotation=30, ha="right")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "06_weekly_timeseries.png"), dpi=150)
plt.close()
print("Saved: 06_weekly_timeseries.png")


# ---------------------------------------------------------------------------
# 8. Figure 6 - Latitude distribution, fire type, day vs night
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

axes[0].hist(df["LATITUDE"], bins=90, color="#7EC8A4",
             edgecolor="white", linewidth=0.2, orientation="horizontal")
axes[0].axhline(0, color="grey", lw=0.8, ls="--")
axes[0].set_ylabel("Latitude (deg)")
axes[0].set_xlabel("Count")
axes[0].set_title("Latitude Distribution")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

type_counts = df["TYPE"].value_counts().sort_index()
type_labels = [TYPE_MAP.get(t, str(t)) for t in type_counts.index]
axes[1].barh(type_labels, type_counts.values / 1e6,
             color=sns.color_palette("Set2", len(type_counts)))
axes[1].set_xlabel("Count (millions)")
axes[1].set_title("Fire Detections by Type")

dn = df.groupby(["MONTH","DAYNIGHT"]).size().unstack(fill_value=0)
dn = dn.div(dn.sum(axis=1), axis=0) * 100
x = np.arange(1, 13)
day_key   = "day"   if "day"   in dn.columns else "D"
night_key = "night" if "night" in dn.columns else "N"
day_pct   = dn.get(day_key,   pd.Series(0, index=dn.index))
night_pct = dn.get(night_key, pd.Series(0, index=dn.index))
axes[2].bar(x, day_pct,   label="Day",   color="#FFD166")
axes[2].bar(x, night_pct, bottom=day_pct, label="Night", color="#073B4C")
axes[2].set_xticks(x)
axes[2].set_xticklabels(MONTH_NAMES, rotation=30, ha="right")
axes[2].set_ylabel("% of monthly detections")
axes[2].set_title("Day vs Night Detections by Month")
axes[2].legend()

plt.suptitle("Spatial & Classification Breakdown - VIIRS 2018", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "07_spatial_classification.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 07_spatial_classification.png")


# ---------------------------------------------------------------------------
# 9. Figure 7 - Correlation heatmap & FRP monthly boxplot
# ---------------------------------------------------------------------------

num_cols = ["LATITUDE","LONGITUDE","BRIGHTNESS","BRIGHT_T31","FRP","SCAN","TRACK","MONTH"]
corr_mat  = df[num_cols].corr()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(
    corr_mat, annot=True, fmt=".2f", cmap="coolwarm",
    center=0, square=True, linewidths=0.5,
    ax=axes[0], annot_kws={"size": 8},
)
axes[0].set_title("Pearson Correlation Matrix")

frp_cap = df["FRP"].quantile(0.95)
sub95 = df[df["FRP"] <= frp_cap]
monthly_frp = [sub95.loc[sub95["MONTH"] == m, "FRP"].values for m in range(1, 13)]
bp = axes[1].boxplot(
    monthly_frp, patch_artist=True, showfliers=False,
    medianprops={"color": "black", "lw": 1.5},
)
colors = plt.cm.RdYlBu_r(np.linspace(0, 1, 12))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(MONTH_NAMES, rotation=30, ha="right")
axes[1].set_xlabel("Month")
axes[1].set_ylabel(f"FRP (MW)  [<=  {frp_cap:.0f} MW, 95th pct]")
axes[1].set_title("Monthly FRP Distribution (boxplot, no outliers)")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "08_correlation_frp_monthly.png"), dpi=150)
plt.close()
print("Saved: 08_correlation_frp_monthly.png")


# ---------------------------------------------------------------------------
# 10. Figure 8 - Top fire events
# ---------------------------------------------------------------------------

top = df.nlargest(20, "FRP")[
    ["ACQ_DATE","LATITUDE","LONGITUDE","FRP","BRIGHTNESS","CONFIDENCE","DAYNIGHT","TYPE"]
].copy()
top["ACQ_DATE"]   = top["ACQ_DATE"].dt.date
top["FRP"]        = top["FRP"].round(1)
top["BRIGHTNESS"] = top["BRIGHTNESS"].round(1)
top.to_csv(os.path.join(OUT_DIR, "09_top20_frp_events.csv"), index=False)

fig, ax = plt.subplots(figsize=(12, 5))
ax.barh(range(len(top)), top["FRP"].values, color="#E63946")
ax.set_yticks(range(len(top)))
ax.set_yticklabels(
    [f"{row.ACQ_DATE}  ({row.LATITUDE:.1f} deg, {row.LONGITUDE:.1f} deg)"
     for row in top.itertuples()],
    fontsize=8,
)
ax.set_xlabel("FRP (MW)")
ax.set_title("Top 20 Fire Events by FRP - VIIRS 2018")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "09_top20_frp_events.png"), dpi=150)
plt.close()
print("Saved: 09_top20_frp_events.png")

print(top.to_string(index=False))


# ---------------------------------------------------------------------------
# 11. Figure 9 - Latitude x Month density heatmap
# ---------------------------------------------------------------------------

lat_bins   = np.arange(-90, 91, 5)
lat_labels = [f"{b:.0f}" for b in lat_bins[:-1]]
df["LAT_BIN"] = pd.cut(df["LATITUDE"], bins=lat_bins, labels=lat_labels, right=False)

heat = (df.groupby(["LAT_BIN","MONTH"])
          .size()
          .unstack(fill_value=0))

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(
    heat / 1e3,
    cmap="YlOrRd", ax=ax,
    cbar_kws={"label": "Detections (thousands)"},
    linewidths=0,
)
ax.set_xlabel("Month")
ax.set_ylabel("Latitude band (deg)")
ax.set_xticklabels(MONTH_NAMES, rotation=0)
ax.set_title("Fire Detection Density: Latitude x Month - VIIRS 2018")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "10_lat_month_heatmap.png"), dpi=150)
plt.close()
print("Saved: 10_lat_month_heatmap.png")


# ---------------------------------------------------------------------------
# 12. Summary report
# ---------------------------------------------------------------------------

peak_month = MONTH_NAMES[int(df["MONTH"].value_counts().idxmax()) - 1]
peak_count = int(df["MONTH"].value_counts().max())

report_lines = [
    "VIIRS ACTIVE FIRE 2018 - EDA SUMMARY",
    "=" * 50,
    f"Total detections              : {len(df):>15,}",
    f"Date range                    : 2018-01-01 to 2018-12-31",
    f"Lat range                     : {df['LATITUDE'].min():.2f} to {df['LATITUDE'].max():.2f}",
    f"Lon range                     : {df['LONGITUDE'].min():.2f} to {df['LONGITUDE'].max():.2f}",
    f"Daytime detections            : {df['DAYNIGHT'].str.lower().str.startswith('d').sum():>15,}  ({df['DAYNIGHT'].str.lower().str.startswith('d').mean()*100:.1f}%)",
    f"Nighttime detections          : {df['DAYNIGHT'].str.lower().str.startswith('n').sum():>15,}  ({df['DAYNIGHT'].str.lower().str.startswith('n').mean()*100:.1f}%)",
    "",
    "FRP (Fire Radiative Power, MW)",
    f"  Mean                        : {df['FRP'].mean():.2f}",
    f"  Median                      : {df['FRP'].median():.2f}",
    f"  Std                         : {df['FRP'].std():.2f}",
    f"  95th pct                    : {df['FRP'].quantile(0.95):.2f}",
    f"  Max                         : {df['FRP'].max():.2f}",
    "",
    "Brightness T21 (K)",
    f"  Mean                        : {df['BRIGHTNESS'].mean():.2f}",
    f"  Median                      : {df['BRIGHTNESS'].median():.2f}",
    f"  Min / Max                   : {df['BRIGHTNESS'].min():.2f} / {df['BRIGHTNESS'].max():.2f}",
    "",
    "Confidence breakdown",
] + [f"  {k:<28}: {v*100:.1f}%"
     for k, v in df["CONFIDENCE"].value_counts(normalize=True).items()] + [
    "",
    "Fire type breakdown",
] + [f"  {TYPE_MAP.get(k, str(k)):<28}: {v*100:.1f}%"
     for k, v in df["TYPE"].value_counts(normalize=True).items()] + [
    "",
    f"Peak fire month (global)      : {peak_month} ({peak_count:,} detections)",
    "",
    "Output files in eda_output/",
    "  01_descriptive_stats.csv",
    "  02_monthly_fire_counts.png",
    "  03_spatial_distribution.png",
    "  04_frp_distribution.png",
    "  05_brightness_temp.png",
    "  06_weekly_timeseries.png",
    "  07_spatial_classification.png",
    "  08_correlation_frp_monthly.png",
    "  09_top20_frp_events.png / .csv",
    "  10_lat_month_heatmap.png",
    "  00_eda_summary.txt",
]

report = "\n".join(report_lines)
print("\n" + report)

with open(os.path.join(OUT_DIR, "00_eda_summary.txt"), "w", encoding="utf-8") as f:
    f.write(report + "\n")

print(f"\nAll outputs saved to: {OUT_DIR}/")
