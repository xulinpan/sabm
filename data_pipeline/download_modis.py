"""
Download script for the MODIS MOD05_L2 precipitable water dataset
used in Rios & Lee (2026) REX-SUB paper.

Product  : MOD05_L2 — MODIS/Terra Total Precipitable Water Vapor
           5-Min L2 Swath 1km and 5km, Collection 6.1
Granule  : MOD05_L2.A2023173.0120.061.2023173131455.hdf
Date     : 2023-06-22 01:20 UTC  (day-of-year 173)
Region   : lon [-57.0, -24.5]  lat [29.9, 51.3]  (western North Atlantic)
N        : 2,748,620 locations at 1km x 1km

Requirements
------------
    pip install earthaccess h5py numpy pandas pyhdf

NASA Earthdata account required (free):
    https://urs.earthdata.nasa.gov/users/new
"""

import os
import sys

# ── Method 1: earthaccess (recommended) ──────────────────────────────────────

def _setup_netrc(username: str, password: str) -> None:
    """Write NASA Earthdata credentials to the Windows _netrc file."""
    netrc_path = os.path.join(os.path.expanduser("~"), "_netrc")
    entry = (
        "machine urs.earthdata.nasa.gov\n"
        f"login {username}\n"
        f"password {password}\n"
    )
    with open(netrc_path, "w") as f:
        f.write(entry)
    print(f"Credentials saved to {netrc_path}")


def download_via_earthaccess(
    outdir: str = ".",
    username: str | None = None,
    password: str | None = None,
) -> list[str]:
    """
    Download the exact granule used in the REX-SUB paper using the
    earthaccess library.

    Authentication priority:
      1. username/password passed directly to this function
      2. EARTHDATA_USERNAME / EARTHDATA_PASSWORD environment variables
      3. Existing ~/._netrc credentials file
      4. Interactive prompt (username + password in terminal)
    """
    try:
        import earthaccess
    except ImportError:
        print("Install earthaccess:  pip install earthaccess")
        sys.exit(1)

    os.makedirs(outdir, exist_ok=True)

    # If credentials provided directly, write _netrc so earthaccess finds them
    if username and password:
        _setup_netrc(username, password)

    # Try environment variables first, then netrc, then interactive prompt
    env_user = os.environ.get("EARTHDATA_USERNAME")
    env_pass = os.environ.get("EARTHDATA_PASSWORD")

    if env_user and env_pass:
        earthaccess.login(strategy="environment")
    else:
        try:
            earthaccess.login(strategy="netrc")
        except Exception:
            print("No saved credentials found — enter your NASA Earthdata login.")
            print("(Register free at https://urs.earthdata.nasa.gov/users/new)")
            earthaccess.login(strategy="interactive")

    # Search for the specific granule
    # DOI: 10.5067/MODIS/MOD05_L2.061
    results = earthaccess.search_data(
        short_name   = "MOD05_L2",
        version      = "061",
        temporal     = ("2023-06-22T01:15:00", "2023-06-22T01:25:00"),
        bounding_box = (-57.0, 29.9, -24.5, 51.3),   # W, S, E, N
    )

    if not results:
        print("[ERROR] No granules found. Check temporal/spatial parameters.")
        return []

    print(f"Found {len(results)} granule(s):")
    for r in results:
        print(f"  {r['meta']['native-id']}")

    # Download
    files = earthaccess.download(results, local_path=outdir)
    print(f"\nDownloaded to: {outdir}")
    return files


# ── Method 2: Direct wget with LAADS App Token ───────────────────────────────

GRANULE_URL = (
    "https://ladsweb.modaps.eosdis.nasa.gov"
    "/archive/allData/61/MOD05_L2/2023/173"
    "/MOD05_L2.A2023173.0120.061.2023173131455.hdf"
)

def download_via_wget(token: str, outdir: str = ".") -> str:
    """
    Download with a LAADS App Token (get yours at:
    https://ladsweb.modaps.eosdis.nasa.gov/profile/#app-keys).

    Usage:
        python download_modis.py --method wget --token YOUR_TOKEN_HERE
    """
    import subprocess

    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, os.path.basename(GRANULE_URL))

    cmd = [
        "wget", "-c", "--no-verbose",
        "--header", f"Authorization: Bearer {token}",
        "-O", outfile,
        GRANULE_URL,
    ]

    print(f"Downloading: {os.path.basename(GRANULE_URL)}")
    result = subprocess.run(cmd, check=True)
    print(f"Saved to: {outfile}")
    return outfile


# ── Method 3: Python requests with token ─────────────────────────────────────

def download_via_requests(token: str, outdir: str = ".") -> str:
    """
    Pure Python download alternative (no wget required).
    Get your LAADS token at:
        https://ladsweb.modaps.eosdis.nasa.gov/profile/#app-keys
    """
    import requests

    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, os.path.basename(GRANULE_URL))

    if os.path.exists(outfile):
        print(f"Already exists: {outfile}")
        return outfile

    print(f"Downloading {os.path.basename(GRANULE_URL)} ...")
    headers = {"Authorization": f"Bearer {token}"}

    with requests.get(GRANULE_URL, headers=headers, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(outfile, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                f.write(chunk)
                downloaded += len(chunk)
                pct = 100 * downloaded / total if total else 0
                print(f"\r  {pct:5.1f}%  ({downloaded/1e6:.1f} MB)", end="")
    print(f"\nSaved: {outfile}")
    return outfile


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download MOD05_L2 granule for REX-SUB paper replication"
    )
    parser.add_argument(
        "--method",
        choices=["earthaccess", "wget", "requests"],
        default="earthaccess",
        help="Download method (default: earthaccess)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="LAADS App Token (required for wget/requests methods). "
             "Get at: https://ladsweb.modaps.eosdis.nasa.gov/profile/#app-keys",
    )
    parser.add_argument(
        "--username",
        default=None,
        help="NASA Earthdata username (earthaccess method). "
             "Alternative: set EARTHDATA_USERNAME env var.",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="NASA Earthdata password (earthaccess method). "
             "Alternative: set EARTHDATA_PASSWORD env var.",
    )
    parser.add_argument(
        "--outdir",
        default="data/raw",
        help="Directory to save the HDF file (default: data/raw)",
    )
    args = parser.parse_args()

    if args.method == "earthaccess":
        download_via_earthaccess(
            outdir=args.outdir,
            username=args.username,
            password=args.password,
        )

    elif args.method in ("wget", "requests"):
        if not args.token:
            print(
                "[ERROR] --token is required for wget/requests method.\n"
                "Get your free LAADS App Token at:\n"
                "  https://ladsweb.modaps.eosdis.nasa.gov/profile/#app-keys"
            )
            sys.exit(1)
        if args.method == "wget":
            download_via_wget(token=args.token, outdir=args.outdir)
        else:
            download_via_requests(token=args.token, outdir=args.outdir)
