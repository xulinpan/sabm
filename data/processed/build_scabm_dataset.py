"""
build_scabm_dataset.py
======================
Processes NEON mosquito surveillance data (UNDE site, 2024) into the
format required by the SC-ABM count-multisource model.

Two-channel observation design exploiting NEON's dual trap-cycle protocol:
  Each biweekly visit runs TWO consecutive ~12-hour trap deployments:
    Overnight trap: set ~7pm CDT, collected ~7am CDT next morning
    Daytime trap:  set ~7am CDT, collected ~7pm CDT same day

  Y^(C)  <- overnight trap count  (high mosquito activity; NegBin)
  Y^(P)  <- daytime trap binary   (lower activity; independent Bernoulli)

Both channels observe the SAME latent N_{i,t} but are conditionally
independent given N, satisfying the SC-ABM dual-source assumption.

Y^(C) count uses proportionIdentified scaling:
  y_C = round(raw_identified_count / proportionIdentified)
  Raw count and prop stored as separate columns for use as offset.

Inputs  (D:/research/data/raw/)
  mos_trapping_stacked.csv
  mos_sorting_stacked.csv
  mos_expertTaxonomistIDProcessed_stacked.csv

Outputs (D:/research/data/processed/)
  mos_grid.csv        -- m rows: cell_idx, lat, lon, elev, NLCD, X/W covariates
  mos_dist_km.csv     -- m x m distance matrix (km)
  mos_neighbors.csv   -- long: cell_idx, neighbor_idx, dist_km
  mos_time_index.csv  -- T rows: time_idx, date_CDT, day_of_year
  mos_obs_C.csv       -- overnight trap: cell_idx, time_idx, y_C, raw_count,
                          prop_identified, trap_hours, trap_type_co2
  mos_obs_P.csv       -- daytime trap:  cell_idx, time_idx, y_P, trap_hours
  mos_summary.json    -- metadata
"""

import csv, collections, json, math, os
from datetime import datetime, timedelta

# ── Configuration ──────────────────────────────────────────────────────────────
SITE          = "UNDE"
FOCAL_SPECIES = "Coquillettidia perturbans"
SEASON_START  = "2024-04-01"
SEASON_END    = "2024-10-31"

# Michigan in summer uses CDT = UTC-5
UTC_OFFSET_H  = -5

# Local CDT hour threshold separating trap cycles:
#   morning collection (overnight trap): local hour in [6, 13]
#   evening collection (daytime trap):   local hour in [17, 23]
MORNING_HOUR_MAX = 14   # local hour < this = morning collection

# Biweekly steps: events within GROUP_DAYS of each other -> same time step
GROUP_DAYS  = 3
# Neighbourhood threshold
NEIGH_KM    = 2.5

RAW = "D:/research/data/raw"
OUT = "D:/research/data/processed"

# ── Helpers ────────────────────────────────────────────────────────────────────
def parse_utc(s):
    """Parse ISO datetime string (with or without trailing Z/seconds)."""
    s = s.rstrip("Z").replace("+00:00", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    # try with seconds stripped to match HH:MM format
    if "T" in s and len(s) > 16:
        try:
            return datetime.strptime(s[:16], "%Y-%m-%dT%H:%M")
        except ValueError:
            pass
    return None

def local_dt(utc_dt):
    """Convert UTC datetime to local CDT."""
    return utc_dt + timedelta(hours=UTC_OFFSET_H)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def standardise(vals):
    mu  = sum(vals) / len(vals)
    var = sum((v - mu)**2 for v in vals) / max(len(vals) - 1, 1)
    sd  = math.sqrt(var) if var > 0 else 1.0
    return [(v - mu) / sd for v in vals], mu, sd

WETLAND_CLASSES = {"woodyWetlands", "emergentHerbaceousWetlands"}
FOREST_CLASSES  = {"evergreenForest", "deciduousForest", "mixedForest"}

# ── Load raw files ─────────────────────────────────────────────────────────────
print("Loading raw files ...")
with open(f"{RAW}/mos_trapping_stacked.csv", encoding="utf-8") as f:
    trap_all = list(csv.DictReader(f))
with open(f"{RAW}/mos_sorting_stacked.csv", encoding="utf-8") as f:
    sort_all = list(csv.DictReader(f))
with open(f"{RAW}/mos_expertTaxonomistIDProcessed_stacked.csv", encoding="utf-8") as f:
    eid_all = list(csv.DictReader(f))

# ── Step 1: active-season trap events at UNDE, parse local datetime ───────────
print(f"Filtering to site={SITE}, season {SEASON_START} - {SEASON_END} ...")

trap_site = []
for r in trap_all:
    if r["siteID"] != SITE or r["samplingImpractical"] != "OK":
        continue
    if not r["collectDate"]:
        continue
    utc = parse_utc(r["collectDate"])
    if utc is None:
        continue
    local = local_dt(utc)
    local_date = local.date().isoformat()
    if not (SEASON_START <= local_date <= SEASON_END):
        continue
    r = dict(r)
    r["_local_dt"]   = local
    r["_local_date"] = local_date
    r["_local_hour"] = local.hour
    r["_channel"]    = "C" if local.hour < MORNING_HOUR_MAX else "P"
    trap_site.append(r)

n_C_events = sum(1 for r in trap_site if r["_channel"] == "C")
n_P_events = sum(1 for r in trap_site if r["_channel"] == "P")
print(f"  Total active events: {len(trap_site)}  "
      f"(overnight/Y_C: {n_C_events}, daytime/Y_P: {n_P_events})")

# ── Step 2: spatial grid ───────────────────────────────────────────────────────
print("Building spatial grid ...")
plot_lats = collections.defaultdict(list)
plot_lons = collections.defaultdict(list)
plot_elev = collections.defaultdict(list)
plot_nlcd = collections.defaultdict(list)

for r in trap_site:
    pid = r["plotID"]
    if r["decimalLatitude"]:
        plot_lats[pid].append(float(r["decimalLatitude"]))
        plot_lons[pid].append(float(r["decimalLongitude"]))
    if r["elevation"]:
        plot_elev[pid].append(float(r["elevation"]))
    if r["nlcdClass"]:
        plot_nlcd[pid].append(r["nlcdClass"])

plots    = sorted(plot_lats.keys())
m        = len(plots)
plot2idx = {p: i for i, p in enumerate(plots)}

lats  = [sum(plot_lats[p]) / len(plot_lats[p]) for p in plots]
lons  = [sum(plot_lons[p]) / len(plot_lons[p]) for p in plots]
elevs = [sum(plot_elev[p]) / len(plot_elev[p]) if plot_elev[p] else 0.0
         for p in plots]
nlcds = [collections.Counter(plot_nlcd[p]).most_common(1)[0][0]
         if plot_nlcd[p] else "unknown" for p in plots]

print(f"  m={m} plots: {plots}")

# ── Step 3: distance matrix and neighbourhood ──────────────────────────────────
print("Computing distances ...")
dist_km = [[haversine_km(lats[i], lons[i], lats[j], lons[j])
            for j in range(m)] for i in range(m)]

neighbours = {i: [j for j in range(m) if j != i and dist_km[i][j] <= NEIGH_KM]
              for i in range(m)}
n_edges = sum(len(v) for v in neighbours.values())
print(f"  Threshold {NEIGH_KM} km -> {n_edges // 2} undirected edges, "
      f"mean {n_edges / m:.1f} neighbours/cell")

# ── Step 4: time steps (based on local CDT date) ──────────────────────────────
print("Building time index ...")
all_local_dates = sorted({r["_local_date"] for r in trap_site})

steps   = []
current = [all_local_dates[0]]
for d in all_local_dates[1:]:
    gap = (datetime.fromisoformat(d) - datetime.fromisoformat(current[-1])).days
    if gap <= GROUP_DAYS:
        current.append(d)
    else:
        steps.append(current)
        current = [d]
steps.append(current)

step_dates = [sorted(s)[0] for s in steps]
date2step  = {d: t for t, step in enumerate(steps) for d in step}
T          = len(steps)

print(f"  {len(all_local_dates)} local dates -> {T} biweekly time steps")
print(f"  Season: {step_dates[0]} to {step_dates[-1]}")

# ── Step 5: subsampleID lookups ────────────────────────────────────────────────
sub2sample = {r["subsampleID"]: r["sampleID"] for r in sort_all}
sub2prop   = {r["subsampleID"]: float(r["proportionIdentified"] or 1.0)
              for r in sort_all}

# ── Step 6: expert ID counts per sampleID for focal species ───────────────────
print(f"Extracting '{FOCAL_SPECIES}' counts ...")

# Per subsample: sum raw counts across sexes then scale
raw_by_sub  = collections.defaultdict(int)   # subsampleID -> raw count
for r in eid_all:
    if r["siteID"] != SITE:
        continue
    if r["scientificName"] != FOCAL_SPECIES:
        continue
    if not r["individualCount"].isdigit():
        continue
    raw_by_sub[r["subsampleID"]] += int(r["individualCount"])

# Per sampleID: sum scaled counts across subsamples
# Each subsample represents proportionIdentified fraction of total catch
# Scaled estimate = raw / prop; sum across subsamples of same sample
focal_scaled = collections.defaultdict(int)   # sampleID -> scaled total
focal_raw    = collections.defaultdict(int)   # sampleID -> raw identified
focal_prop   = collections.defaultdict(float) # sampleID -> representative prop

for sub, raw in raw_by_sub.items():
    sid  = sub2sample.get(sub, "")
    prop = sub2prop.get(sub, 1.0)
    if not sid:
        continue
    scaled = round(raw / prop) if prop > 0 else raw
    focal_scaled[sid] += scaled
    focal_raw[sid]    += raw
    # Track representative prop (use minimum = largest subsample = most weight)
    if sid not in focal_prop or prop < focal_prop[sid]:
        focal_prop[sid] = prop

print(f"  Samples with focal species: {len(focal_scaled)}")
print(f"  Total scaled individuals:   {sum(focal_scaled.values()):,}")
print(f"  Total raw identified:       {sum(focal_raw.values()):,}")

# ── Step 7: build observation tables ──────────────────────────────────────────
print("Assembling observation tables ...")

rows_C = []  # overnight: (cell_idx, time_idx, y_C, raw, prop, trap_hours, trap_type_co2)
rows_P = []  # daytime:   (cell_idx, time_idx, y_P, trap_hours)

for r in trap_site:
    pid = r["plotID"]
    if pid not in plot2idx:
        continue
    ci   = plot2idx[pid]
    ti   = date2step.get(r["_local_date"])
    if ti is None:
        continue

    trap_hours = float(r["trapHours"]) if r["trapHours"] else 0.0
    is_co2     = 1 if r["trapType"] in ("CO2 canister", "dry ice") else 0
    sid        = r["sampleID"]

    if r["_channel"] == "C":
        # Overnight trap -> Y^(C)
        # Need a collected sample (sampleID non-empty) to get counts
        if sid:
            y_C   = focal_scaled.get(sid, 0)
            raw   = focal_raw.get(sid, 0)
            prop  = focal_prop.get(sid, 1.0)
            rows_C.append((ci, ti, y_C, raw, round(prop, 4), trap_hours, is_co2))

    else:
        # Daytime trap -> Y^(P) (binary only, no expert species ID needed)
        # Use targetTaxaPresent: Y=any mosquito caught, N=empty trap
        ttp = r["targetTaxaPresent"]
        if ttp in ("Y", "N"):
            y_P = 1 if ttp == "Y" else 0
            rows_P.append((ci, ti, y_P, trap_hours))

# Deduplicate (cell, time) collisions within a biweekly step:
# Y^(C): take the observation with the largest scaled count
# Y^(P): presence wins (OR aggregation)
def dedup_C(rows):
    best = {}
    for ci, ti, y_C, raw, prop, th, tt in rows:
        key = (ci, ti)
        if key not in best or y_C > best[key][2]:
            best[key] = (ci, ti, y_C, raw, prop, th, tt)
    return sorted(best.values())

def dedup_P(rows):
    best = {}
    for ci, ti, y_P, th in rows:
        key = (ci, ti)
        if key not in best or y_P > best[key][2]:
            best[key] = (ci, ti, y_P, th)
    return sorted(best.values())

rows_C = dedup_C(rows_C)
rows_P = dedup_P(rows_P)

nC       = len(rows_C)
nC_pos   = sum(1 for r in rows_C if r[2] > 0)
nC_zero  = nC - nC_pos
nP       = len(rows_P)
nP_pos   = sum(1 for r in rows_P if r[2] > 0)
nP_abs   = nP - nP_pos

print(f"  O_C (overnight): {nC} entries  ({nC_pos} positive, {nC_zero} zeros)  "
      f"coverage {nC / (m * T) * 100:.1f}%")
print(f"  O_P (daytime):   {nP} entries  ({nP_pos} presence,  {nP_abs} absence) "
      f"coverage {nP / (m * T) * 100:.1f}%")

# count stats (positive overnight traps only)
pos_counts = [r[2] for r in rows_C if r[2] > 0]
print(f"  Y^C stats: min={min(pos_counts)}  max={max(pos_counts)}  "
      f"mean={sum(pos_counts)/len(pos_counts):.0f}  "
      f"median={sorted(pos_counts)[len(pos_counts)//2]}")

# ── Step 8: covariate matrices ─────────────────────────────────────────────────
print("Building covariate matrices ...")
elev_z, elev_mu, elev_sd = standardise(elevs)

# X (m x 4): GP prior covariates for habitat suitability alpha_i
X_mat = []
for i in range(m):
    is_wet    = 1.0 if nlcds[i] in WETLAND_CLASSES else 0.0
    is_forest = 1.0 if nlcds[i] in FOREST_CLASSES  else 0.0
    X_mat.append([1.0, round(elev_z[i], 6), is_wet, is_forest])

# W (m x 2): covariates for log(K_i) and LDD log(psi_i)
W_mat = [[1.0, round(elev_z[i], 6)] for i in range(m)]

# ── Step 9: write outputs ──────────────────────────────────────────────────────
print("Writing output files ...")

with open(f"{OUT}/mos_grid.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["cell_idx", "plot_id", "lat", "lon", "elevation_m",
                "nlcd_class", "is_wetland", "is_forest",
                "X_intercept", "X_elev_z", "X_wetland", "X_forest",
                "W_intercept", "W_elev_z"])
    for i in range(m):
        w.writerow([i, plots[i],
                    round(lats[i], 6), round(lons[i], 6), round(elevs[i], 1),
                    nlcds[i],
                    int(X_mat[i][2]), int(X_mat[i][3]),
                    *X_mat[i], *W_mat[i]])

with open(f"{OUT}/mos_dist_km.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([""] + list(range(m)))
    for i in range(m):
        w.writerow([i] + [round(dist_km[i][j], 4) for j in range(m)])

with open(f"{OUT}/mos_neighbors.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["cell_idx", "neighbor_idx", "dist_km"])
    for i in range(m):
        for j in neighbours[i]:
            w.writerow([i, j, round(dist_km[i][j], 4)])

with open(f"{OUT}/mos_time_index.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["time_idx", "date_CDT", "day_of_year", "dates_in_step"])
    for t, (date, step) in enumerate(zip(step_dates, steps)):
        doy = datetime.fromisoformat(date).timetuple().tm_yday
        w.writerow([t, date, doy, ";".join(sorted(step))])

with open(f"{OUT}/mos_obs_C.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["cell_idx", "time_idx", "y_C", "raw_count",
                "prop_identified", "trap_hours", "trap_type_co2"])
    for row in rows_C:
        w.writerow([row[0], row[1], row[2], row[3], row[4],
                    round(row[5], 2), row[6]])

with open(f"{OUT}/mos_obs_P.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["cell_idx", "time_idx", "y_P", "trap_hours"])
    for row in rows_P:
        w.writerow([row[0], row[1], row[2], round(row[3], 2)])

summary = {
    "site":          SITE,
    "focal_species": FOCAL_SPECIES,
    "season":        f"{SEASON_START} to {SEASON_END}",
    "m":             m,
    "T":             T,
    "plots":         plots,
    "step_dates":    step_dates,
    "channel_design": {
        "Y_C": "overnight trap (set ~7pm CDT, collected ~7am CDT): NegBin count",
        "Y_P": "daytime trap  (set ~7am CDT, collected ~7pm CDT): Bernoulli binary",
        "independence": "different trap cycles, conditionally independent given N",
    },
    "neigh_threshold_km": NEIGH_KM,
    "observation": {
        "n_OC": nC, "n_OC_pos": nC_pos, "n_OC_zero": nC_zero,
        "n_OP": nP, "n_OP_pos": nP_pos, "n_OP_abs": nP_abs,
        "coverage_C_pct": round(nC / (m * T) * 100, 1),
        "coverage_P_pct": round(nP / (m * T) * 100, 1),
    },
    "counts": {
        "total_scaled":  sum(r[2] for r in rows_C),
        "min":    min(pos_counts),
        "max":    max(pos_counts),
        "mean":   round(sum(pos_counts) / len(pos_counts), 1),
        "median": sorted(pos_counts)[len(pos_counts) // 2],
    },
    "covariates": {
        "X_cols": ["intercept", "elevation_z", "is_wetland", "is_forest"],
        "W_cols": ["intercept", "elevation_z"],
        "elevation_mu_m": round(elev_mu, 1),
        "elevation_sd_m": round(elev_sd, 1),
        "nlcd_classes":   nlcds,
    },
}
with open(f"{OUT}/mos_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n=== DONE ===")
print(f"  Site: {SITE}  |  Species: {FOCAL_SPECIES}")
print(f"  m={m}, T={T},  season: {step_dates[0]} to {step_dates[-1]}")
print(f"  O_C: {nC} entries ({nC_pos} pos / {nC_zero} zeros),  coverage {nC/(m*T)*100:.0f}%")
print(f"  O_P: {nP} entries ({nP_pos} pres / {nP_abs} abs),  coverage {nP/(m*T)*100:.0f}%")
print(f"  Y^C: min={min(pos_counts)}  max={max(pos_counts)}  mean={sum(pos_counts)/len(pos_counts):.0f}  "
      f"median={sorted(pos_counts)[len(pos_counts)//2]}")
print()
print("Output files:")
for fname in ["mos_grid.csv", "mos_dist_km.csv", "mos_neighbors.csv",
              "mos_time_index.csv", "mos_obs_C.csv", "mos_obs_P.csv",
              "mos_summary.json"]:
    size = os.path.getsize(f"{OUT}/{fname}")
    print(f"  {fname:<30}  {size:>8,} bytes")
