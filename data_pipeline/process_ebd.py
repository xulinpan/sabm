"""
Process the eBird Basic Dataset (EBD) sample zip file.

Input  : data/raw/ebd-datafile-SAMPLE.zip
         - ebd_US-AL-125_202503_202503_smp_relMar-2025.txt      (observations)
         - ebd_US-AL-125_202503_202503_smp_relMar-2025_sampling.txt (checklists)

Outputs: data/processed/
         ebd_observations.parquet   — cleaned observation records
         ebd_sampling.parquet       — cleaned sampling / checklist events
         ebd_species_summary.csv    — per-species aggregate statistics
         ebd_checklist_summary.csv  — per-checklist species richness
"""

import io
import os
import zipfile
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────

ZIP_PATH = "data/raw/ebd-datafile-SAMPLE.zip"
OUT_DIR  = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

EBD_FILE = "ebd_US-AL-125_202503_202503_smp_relMar-2025.txt"
SMP_FILE = "ebd_US-AL-125_202503_202503_smp_relMar-2025_sampling.txt"

# Columns to keep from the observation file
OBS_KEEP = [
    "GLOBAL UNIQUE IDENTIFIER",
    "TAXONOMIC ORDER",
    "CATEGORY",
    "COMMON NAME",
    "SCIENTIFIC NAME",
    "EXOTIC CODE",
    "OBSERVATION COUNT",
    "BREEDING CODE",
    "BREEDING CATEGORY",
    "BEHAVIOR CODE",
    "COUNTRY",
    "STATE",
    "COUNTY",
    "COUNTY CODE",
    "LOCALITY",
    "LOCALITY ID",
    "LOCALITY TYPE",
    "LATITUDE",
    "LONGITUDE",
    "OBSERVATION DATE",
    "TIME OBSERVATIONS STARTED",
    "OBSERVER ID",
    "SAMPLING EVENT IDENTIFIER",
    "PROTOCOL NAME",
    "PROTOCOL CODE",
    "DURATION MINUTES",
    "EFFORT DISTANCE KM",
    "NUMBER OBSERVERS",
    "ALL SPECIES REPORTED",
    "GROUP IDENTIFIER",
    "HAS MEDIA",
    "APPROVED",
    "REVIEWED",
    "CHECKLIST COMMENTS",
    "SPECIES COMMENTS",
]

SMP_KEEP = [
    "SAMPLING EVENT IDENTIFIER",
    "COUNTRY",
    "STATE",
    "COUNTY",
    "COUNTY CODE",
    "LOCALITY",
    "LOCALITY ID",
    "LOCALITY TYPE",
    "LATITUDE",
    "LONGITUDE",
    "OBSERVATION DATE",
    "TIME OBSERVATIONS STARTED",
    "OBSERVER ID",
    "PROTOCOL NAME",
    "PROTOCOL CODE",
    "DURATION MINUTES",
    "EFFORT DISTANCE KM",
    "NUMBER OBSERVERS",
    "ALL SPECIES REPORTED",
    "GROUP IDENTIFIER",
]


# ── 1. Load raw files ─────────────────────────────────────────────────────────

print(f"Reading {ZIP_PATH} ...")
with zipfile.ZipFile(ZIP_PATH) as z:
    raw_obs = z.read(EBD_FILE)
    raw_smp = z.read(SMP_FILE)

obs_raw = pd.read_csv(
    io.StringIO(raw_obs.decode("utf-8")),
    sep="\t", low_memory=False, on_bad_lines="skip",
)
smp_raw = pd.read_csv(
    io.StringIO(raw_smp.decode("utf-8")),
    sep="\t", low_memory=False, on_bad_lines="skip",
)

print(f"  Observations raw : {obs_raw.shape}")
print(f"  Sampling raw     : {smp_raw.shape}")


# ── 2. Clean observations ─────────────────────────────────────────────────────

obs = obs_raw[[c for c in OBS_KEEP if c in obs_raw.columns]].copy()

# Keep only approved records
obs = obs[obs["APPROVED"] == 1].copy()

# Parse date / time
obs["OBSERVATION DATE"] = pd.to_datetime(obs["OBSERVATION DATE"])
obs["MONTH"]  = obs["OBSERVATION DATE"].dt.month
obs["DOW"]    = obs["OBSERVATION DATE"].dt.day_name()
obs["HOUR"]   = (
    obs["TIME OBSERVATIONS STARTED"]
    .str.split(":")
    .str[0]
    .astype(float, errors="ignore")
)

# OBSERVATION COUNT: "X" means presence-only (no count). Convert to NaN for numerics.
obs["COUNT_RAW"] = obs["OBSERVATION COUNT"]
obs["OBSERVATION COUNT"] = pd.to_numeric(
    obs["OBSERVATION COUNT"].replace("X", np.nan), errors="coerce"
)

# Normalise column names: lowercase, underscores
obs.columns = [c.lower().replace(" ", "_").replace("/", "_") for c in obs.columns]

# Deduplicate via GROUP IDENTIFIER (shared checklists): keep one row per
# (species + group) so we don't count the same observation twice.
grp_mask = obs["group_identifier"].notna()
obs_dedup = pd.concat([
    obs[~grp_mask],
    obs[grp_mask].drop_duplicates(subset=["scientific_name", "group_identifier"]),
], ignore_index=True)

print(f"\nObservations after approval filter  : {len(obs):,}")
print(f"Observations after group dedup       : {len(obs_dedup):,}")


# ── 3. Clean sampling ─────────────────────────────────────────────────────────

smp = smp_raw[[c for c in SMP_KEEP if c in smp_raw.columns]].copy()

smp["OBSERVATION DATE"] = pd.to_datetime(smp["OBSERVATION DATE"])
smp["HOUR"] = (
    smp["TIME OBSERVATIONS STARTED"]
    .str.split(":")
    .str[0]
    .astype(float, errors="ignore")
)

smp.columns = [c.lower().replace(" ", "_").replace("/", "_") for c in smp.columns]

# Deduplicate shared checklists in sampling file
grp_smp = smp["group_identifier"].notna()
smp_dedup = pd.concat([
    smp[~grp_smp],
    smp[grp_smp].drop_duplicates(subset=["sampling_event_identifier"]),
], ignore_index=True)

print(f"Checklists raw                       : {len(smp):,}")
print(f"Checklists after group dedup         : {len(smp_dedup):,}")


# ── 4. Per-checklist species richness ─────────────────────────────────────────

# Only complete checklists (all_species_reported = 1, species category only)
complete_obs = obs_dedup[
    (obs_dedup["all_species_reported"] == 1) &
    (obs_dedup["category"] == "species")
]

richness = (
    complete_obs
    .groupby("sampling_event_identifier")
    .agg(
        species_richness   = ("scientific_name", "nunique"),
        total_individuals  = ("observation_count", "sum"),
        has_media          = ("has_media", "max"),
        observer_id        = ("observer_id", "first"),
        latitude           = ("latitude",   "first"),
        longitude          = ("longitude",  "first"),
        observation_date   = ("observation_date", "first"),
        protocol_name      = ("protocol_name", "first"),
        duration_minutes   = ("duration_minutes", "first"),
        effort_distance_km = ("effort_distance_km", "first"),
        number_observers   = ("number_observers", "first"),
    )
    .reset_index()
)

print(f"\nComplete checklists (all-species, species-only): {len(richness):,}")
print(f"  Median species richness  : {richness['species_richness'].median():.0f}")
print(f"  Max species richness     : {richness['species_richness'].max()}")
print(f"  Mean duration (min)      : {richness['duration_minutes'].mean():.1f}")


# ── 5. Per-species summary ────────────────────────────────────────────────────

species_summary = (
    complete_obs
    .groupby(["scientific_name", "common_name"])
    .agg(
        taxonomic_order    = ("taxonomic_order", "first"),
        n_checklists       = ("sampling_event_identifier", "nunique"),
        total_count        = ("observation_count", "sum"),
        max_count          = ("observation_count", "max"),
        has_breeding_code  = ("breeding_code", lambda x: x.notna().any()),
        has_media          = ("has_media", "max"),
        n_localities       = ("locality_id", "nunique"),
    )
    .reset_index()
    .sort_values("n_checklists", ascending=False)
)

# Frequency: fraction of complete checklists the species appeared on
n_complete = richness["sampling_event_identifier"].nunique()
species_summary["frequency"] = (
    species_summary["n_checklists"] / n_complete
).round(4)

print(f"\nUnique species (complete checklists) : {len(species_summary):,}")
print(f"Total complete checklists used       : {n_complete:,}")
print("\nTop 20 most-reported species:")
print(
    species_summary[["common_name","n_checklists","frequency","total_count","max_count"]]
    .head(20)
    .to_string(index=False)
)


# ── 6. Save outputs ───────────────────────────────────────────────────────────

obs_path     = os.path.join(OUT_DIR, "ebd_observations.parquet")
smp_path     = os.path.join(OUT_DIR, "ebd_sampling.parquet")
spp_path     = os.path.join(OUT_DIR, "ebd_species_summary.csv")
ckl_path     = os.path.join(OUT_DIR, "ebd_checklist_summary.csv")

obs_dedup.to_parquet(obs_path, index=False)
smp_dedup.to_parquet(smp_path, index=False)
species_summary.to_csv(spp_path, index=False)
richness.to_csv(ckl_path, index=False)

print(f"\nSaved:")
print(f"  {obs_path:<55}  {len(obs_dedup):>6,} rows")
print(f"  {smp_path:<55}  {len(smp_dedup):>6,} rows")
print(f"  {spp_path:<55}  {len(species_summary):>6,} rows")
print(f"  {ckl_path:<55}  {len(richness):>6,} rows")


# ── 7. Quick validation ───────────────────────────────────────────────────────

print("\n=== Validation ===")
print(f"All approved (obs)          : {(obs_dedup['approved'] == 1).all()}")
print(f"No duplicate GUIDs          : {obs_dedup['global_unique_identifier'].nunique() == len(obs_dedup)}")
print(f"Obs date range              : {obs_dedup['observation_date'].min().date()} to "
      f"{obs_dedup['observation_date'].max().date()}")
print(f"County codes                : {sorted(obs_dedup['county_code'].unique())}")
print(f"Protocol types              : {sorted(obs_dedup['protocol_name'].dropna().unique())}")
print(f"Locality types              : {obs_dedup['locality_type'].value_counts().to_dict()}")
print(f"Category breakdown          : {obs_dedup['category'].value_counts().to_dict()}")
print(f"Presence-only (X) records   : {obs_dedup['count_raw'].eq('X').sum():,}")
print()
print("Done.")
