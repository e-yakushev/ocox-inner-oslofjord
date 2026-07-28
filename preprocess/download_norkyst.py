#!/usr/bin/env python3

import argparse
import os
import time
from pathlib import Path

import xarray as xr

from oslofjord_sim.stuff import list_opendap_files

OPENDAP_URL = "https://thredds.met.no/thredds/dodsC/fou-hi/norkyst800m/"
PARAMETERS = ["temperature", "salinity", "u_eastward", "v_northward"]
LATITUDE_RANGE = (58, 60)
LONGITUDE_RANGE = (9, 11)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and combine NorKyst-800m monthly data for an entire year."
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2020,
        help="Year to download, e.g. 2020",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "src" / "oslofjord-sim" / "data" / "input" / "oslofjord",
        help="Output directory for NetCDF files",
    )
    return parser.parse_args()


def process_month(year: int, month: int, output_dir: Path) -> None:
    month_string = f".{year}{month:02d}"
    output_path = output_dir / f"NorKyst-800m_ZDEPTHS_avg_{month_string[1:]}.nc"

    if output_path.exists():
        print(f"✔ Skipping {month_string[1:]} (already exists)")
        return

    print(f"▶ Processing {month_string[1:]}...")

    files = sorted(s for s in list_opendap_files() if month_string in s)

    if not files:
        print(f"⚠ No files found for {month_string[1:]}, skipping.")
        return

    urls = [os.path.join(OPENDAP_URL, x) for x in files]

    print(f"  Opening {len(urls)} datasets...")
    dss = []
    mask = None
    for url in urls:
        ds = None
        for attempt in range(5):
            try:
                ds = xr.open_dataset(url)[PARAMETERS]
                break
            except OSError as e:
                wait = 2 ** attempt
                print(f"    ⚠ Attempt {attempt + 1} failed for {url}: {e}")
                print(f"      Retrying in {wait}s...")
                time.sleep(wait)
        if ds is None:
            print(f"    ✗ Skipping {url} after 5 failed attempts.")
            continue
        if mask is None:
            mask = (
                (ds.lat >= LATITUDE_RANGE[0]) & (ds.lat <= LATITUDE_RANGE[1]) &
                (ds.lon >= LONGITUDE_RANGE[0])  & (ds.lon <= LONGITUDE_RANGE[1])
            )
        ds = ds.where(mask, drop=True)
        dss.append(ds)
        print(f"    Opened: {url}")

    print("  Combining datasets ...")
    ds = xr.combine_by_coords(dss, compat="no_conflicts", combine_attrs="override", coords="different")

    encoding = {var: {"zlib": True, "complevel": 5} for var in ds.data_vars}

    print(f"  Writing output to: {output_path}")
    ds.to_netcdf(output_path, encoding=encoding)

    print(f"✔ Finished {month_string[1:]}\n")


def main() -> None:
    args = parse_args()
    year = args.year
    output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing year: {year}")
    print(f"Output directory: {output_dir}\n")

    for month in range(1, 13):
        process_month(year, month, output_dir)

    print("All done.")


if __name__ == "__main__":
    main()
