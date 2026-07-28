import xml.etree.ElementTree as ET

import numpy as np
import requests
import xarray as xr
import xesmf as xe
from scipy import ndimage
from scipy.interpolate import interp1d

CATALOG_URL = "https://thredds.met.no/thredds/catalog/fou-hi/norkyst800m/catalog.xml"


def tranform_to_z(ds):
    """
    Transforms s coordingate to z with Vtransform = 2
    """
    zo_rho = (ds.hc * ds.s_rho + ds.Cs_r * ds.h) / (ds.hc + ds.h)
    z_rho = ds.zeta + (ds.zeta + ds.h) * zo_rho
    return z_rho.transpose()


def fill_nans_with_nearest(da):
    """Fill NaNs in an xarray DataArray with nearest non-NaN values."""

    # mask: 1 where data is NaN, 0 elsewhere
    nan_mask = np.isnan(da.values)

    # distance transform returns:
    #  - distances
    #  - indices of nearest non-NaN pixel along each axis
    dist, idx = ndimage.distance_transform_edt(nan_mask, return_indices=True)

    # use advanced indexing to build the filled array
    filled = da.values[tuple(idx)]

    return filled


def regrid_depths(values, depths, target_depths):
    """
    Args:
        values: values to be interpolated
        depths: depths of original values
        target_depths: interpolation target depths
    Returns:
        interpolated_values: result values on target depths
    """
    interpolated_shape = list(depths.shape)
    interpolated_shape[1] = len(target_depths)
    interpolated_values = np.empty(interpolated_shape)

    T, D, X, Y = values.shape
    for t in range(T):
        for x in range(X):
            for y in range(Y):
                f = interp1d(
                    depths[t, :, x, y],
                    values[t, :, x, y],
                    kind="linear",
                    bounds_error=False,
                )
                interpolated_values[t, :, x, y] = f(target_depths)

    return interpolated_values


def regrid_from_norkyst(regridder_rho, regridder_u, regridder_v, ds_in, ds_out_c, ds_out_u, ds_out_v, target_depths):
    """
    Regrids norkyst output file from
    https://thredds.met.no/thredds/catalog/fou-hi/norkyst800m/catalog.html
    'ds_in' to target lons, lats, depths.
    Target lons and lats are in ds_out_*, for example

    ds_out_c = xr.Dataset(
        {
            "lat": (["lat"], np.linspace(59.1, 59.98, num=490), {"units": "degrees_north"}),
            "lon": (["lon"], np.linspace(10.2, 10.85, num=88), {"units": "degrees_east"}),
        }
    )
    ds_out_u = xr.Dataset(
        {
            "lat": (["lat"], np.linspace(59.1, 59.98, num=490), {"units": "degrees_north"}),
            "lon": (["lon"], np.linspace(10.2, 10.85, num=88 + 1), {"units": "degrees_east"}),
        }
    )
    ds_out_v = xr.Dataset(
        {
            "lat": (["lat"], np.linspace(59.1, 59.98, num=490 + 1), {"units": "degrees_north"}),
            "lon": (["lon"], np.linspace(10.2, 10.85, num=88), {"units": "degrees_east"}),
        }
    )

    """
    if regridder_rho is None:
        regridder_rho = xe.Regridder(ds_in, ds_out_c, "bilinear", unmapped_to_nan=True)
    if regridder_u is None:
        regridder_u = xe.Regridder(ds_in, ds_out_u, "bilinear", unmapped_to_nan=True)
    if regridder_v is None:
        regridder_v = xe.Regridder(ds_in, ds_out_v, "bilinear", unmapped_to_nan=True)

    da_temp = regridder_rho(ds_in["temperature"])
    da_salt = regridder_rho(ds_in["salinity"])
    da_u = regridder_u(ds_in["u_eastward"])
    da_v = regridder_v(ds_in["v_northward"])

    depths = ds_in.depth.values
    f = interp1d(
        -1 * depths,
        da_temp.values,
        axis=1,
        kind="linear",
        bounds_error=False,
    )
    np_temp = f(target_depths)
    f = interp1d(
        -1 * depths,
        da_salt.values,
        axis=1,
        kind="linear",
        bounds_error=False,
    )
    np_salt = f(target_depths)
    f = interp1d(
        -1 * depths,
        da_u.values,
        axis=1,
        kind="linear",
        bounds_error=False,
    )
    np_u = f(target_depths)
    f = interp1d(
        -1 * depths,
        da_v.values,
        axis=1,
        kind="linear",
        bounds_error=False,
    )
    np_v = f(target_depths)

    np_time = ds_in.time.values

    return regridder_rho, regridder_u, regridder_v, np_time, np_temp, np_salt, np_u, np_v


def regrid_from_roms(ds_in, ds_out_c, ds_out_u, ds_out_v, target_depths):
    """
    Regrids roms output file 'ds_in' to target lons, lats, depths.
    Target lons and lats are in ds_out_*, for example

    ds_out_c = xr.Dataset(
        {
            "lat": (["lat"], np.linspace(59.1, 59.98, num=490), {"units": "degrees_north"}),
            "lon": (["lon"], np.linspace(10.2, 10.85, num=88), {"units": "degrees_east"}),
        }
    )
    ds_out_u = xr.Dataset(
        {
            "lat": (["lat"], np.linspace(59.1, 59.98, num=490), {"units": "degrees_north"}),
            "lon": (["lon"], np.linspace(10.2, 10.85, num=88 + 1), {"units": "degrees_east"}),
        }
    )
    ds_out_v = xr.Dataset(
        {
            "lat": (["lat"], np.linspace(59.1, 59.98, num=490 + 1), {"units": "degrees_north"}),
            "lon": (["lon"], np.linspace(10.2, 10.85, num=88), {"units": "degrees_east"}),
        }
    )

    """
    ds_in["z_rho"] = tranform_to_z(ds_in)

    regridder_rho = xe.Regridder(
        ds_in.rename({"lon_rho": "lon", "lat_rho": "lat"}), ds_out_c, "bilinear", unmapped_to_nan=True
    )
    regridder_u = xe.Regridder(
        ds_in.rename({"lon_u": "lon", "lat_u": "lat"}), ds_out_u, "bilinear", unmapped_to_nan=True
    )
    regridder_v = xe.Regridder(
        ds_in.rename({"lon_v": "lon", "lat_v": "lat"}), ds_out_v, "bilinear", unmapped_to_nan=True
    )

    da_temp = regridder_rho(ds_in["temp"])
    da_salt = regridder_rho(ds_in["salt"])
    da_zrho = regridder_rho(ds_in["z_rho"])
    da_u = regridder_u(ds_in["u"])
    da_v = regridder_v(ds_in["v"])

    zrho = da_zrho.values
    zrho = np.transpose(zrho, (1, 0, 2, 3))

    np_temp = regrid_depths(da_temp.values, zrho, target_depths)
    np_salt = regrid_depths(da_salt.values, zrho, target_depths)

    zu = np.zeros_like(da_u)
    zu[:, :, :, :-1] = zrho
    zu[:, :, :, -1] = zu[:, :, :, -2]
    zv = np.zeros_like(da_v)
    zv[:, :, :-1, :] = zrho
    zv[:, :, -1, :] = zv[:, :, -2, :]

    np_u = regrid_depths(da_u.values, zu, target_depths)
    np_v = regrid_depths(da_v.values, zv, target_depths)

    np_time = ds_in.ocean_time.values

    return np_time, np_temp, np_salt, np_u, np_v


def fill_surrounded_nans(arr):
    # Make a copy to avoid modifying original array
    result = arr.copy()
    rows, cols = arr.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if np.isnan(arr[i, j]):
                west = arr[i, j - 1]
                north = arr[i - 1, j]
                east = arr[i, j + 1]
                south = arr[i + 1, j]

                # Check if all 4 neighbors are NOT NaN
                if not np.isnan(west) and not np.isnan(north) and not np.isnan(east) and not np.isnan(south):
                    result[i, j] = np.mean([west, north, east, south])

    return result


def replace_surrounded_values(arr, sides=3):
    # Create a copy of the array to modify
    new_arr = arr.copy()

    # Get the shape of the array
    rows, cols = arr.shape

    # Iterate through the array (excluding edges to avoid index errors)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if not np.isnan(arr[i, j]):  # Only check non-NaN values
                # Count NaN neighbors
                neighbors = [
                    np.isnan(arr[i - 1, j]) if i > 0 else False,  # Top
                    np.isnan(arr[i + 1, j]) if i < rows - 1 else False,  # Bottom
                    np.isnan(arr[i, j - 1]) if j > 0 else False,  # Left
                    np.isnan(arr[i, j + 1]) if j < cols - 1 else False,  # Right
                ]
                if sum(neighbors) >= sides:
                    new_arr[i, j] = np.nan  # Replace with NaN if surrounded on 3+ sides

    return new_arr


def fill_diagonal_pairs(arr):
    a = arr.copy()

    # Extract all 2×2 blocks using slicing
    tl = a[:-1, :-1]  # top-left
    tr = a[:-1, 1:]  # top-right
    bl = a[1:, :-1]  # bottom-left
    br = a[1:, 1:]  # bottom-right

    # Condition:
    # TL and BR are floats, TR and BL are NaN
    cond = (~np.isnan(tl)) & (~np.isnan(br)) & (np.isnan(tr)) & (np.isnan(bl))

    # Compute fill value
    fill_val = (tl + br) / 2.0

    # Apply to the upper-right cell of each 2×2 block
    a[:-1, 1:][cond] = fill_val[cond]

    return a


def fill_secondary_diagonal_pairs(arr):
    a = arr.copy()

    # Extract all 2×2 slices
    tl = a[:-1, :-1]  # top-left
    tr = a[:-1, 1:]  # top-right
    bl = a[1:, :-1]  # bottom-left
    br = a[1:, 1:]  # bottom-right

    # Condition for secondary diagonal:
    # TR and BL are floats, TL and BR are NaN
    cond = (~np.isnan(tr)) & (~np.isnan(bl)) & (np.isnan(tl)) & (np.isnan(br))

    # Value to insert = average of (TR, BL)
    fill_val = (tr + bl) / 2.0

    # Fill the TOP-LEFT cell of each matching block
    a[:-1, :-1][cond] = fill_val[cond]

    return a


def replace_surrounded_and_clusters(arr, cluster=1):
    new_arr = arr.copy()
    rows, cols = arr.shape

    # First pass: Replace values surrounded by NaNs on at least 3 sides
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if not np.isnan(arr[i, j]):
                # Check top, bottom, left, right
                neighbors = [
                    np.isnan(arr[i - 1, j]) if i > 0 else False,  # Top
                    np.isnan(arr[i + 1, j]) if i < rows - 1 else False,  # Bottom
                    np.isnan(arr[i, j - 1]) if j > 0 else False,  # Left
                    np.isnan(arr[i, j + 1]) if j < cols - 1 else False,  # Right
                ]
                if sum(neighbors) >= 3:
                    new_arr[i, j] = np.nan  # Replace with NaN if surrounded on 3+ sides

    # Second pass: Replace small clusters (≤3 consecutive values) surrounded by NaNs
    def check_and_replace_clusters(arr, axis):
        """Find and replace small clusters of non-NaNs surrounded by NaNs along the given axis."""
        arr = arr.T if axis == 0 else arr  # Transpose if checking vertically

        for i in range(arr.shape[0]):  # Iterate through rows (or columns if transposed)
            row = arr[i]
            nan_mask = np.isnan(row)
            j = 0

            while j < len(row):
                # Find the start of a cluster of non-NaNs
                if not nan_mask[j]:
                    start = j
                    while j < len(row) and not nan_mask[j]:
                        j += 1
                    end = j  # End of cluster (exclusive)

                    # If cluster is 3 or fewer elements and surrounded by NaNs, replace with NaNs
                    if (end - start) <= cluster:
                        left_nan = start == 0 or nan_mask[start - 1]
                        right_nan = end == len(row) or nan_mask[end]
                        if left_nan and right_nan:
                            row[start:end] = np.nan  # Replace the cluster

                j += 1  # Move to the next element

            arr[i] = row  # Update the row in the array

        return arr.T if axis == 0 else arr  # Transpose back if needed

    new_arr = check_and_replace_clusters(new_arr, axis=1)  # Horizontal check
    new_arr = check_and_replace_clusters(new_arr, axis=0)  # Vertical check

    return new_arr


def list_opendap_files(base_url=CATALOG_URL):
    response = requests.get(base_url, timeout=30)

    if response.status_code != 200:
        print("Failed to retrieve the directory listing.")
        return []

    root = ET.fromstring(response.content)
    files = [
        elem.attrib["name"]
        for elem in root.findall(".//{http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0}dataset")
        if elem.attrib["name"].endswith(".nc")
    ]

    return files


def reformat_river_dataset(ds_rivers):
    river_values = ds_rivers.river.values
    properties = list(ds_rivers.data_vars)
    river_list = []
    for river_val in river_values:
        # Select data for this specific river
        river_data = ds_rivers.sel(river=river_val)

        # Stack all data variables into a single array along the 'properties' dimension
        river_arrays = []
        for var_name in properties:
            # Drop the 'river' coordinate to avoid conflicts
            river_arrays.append(river_data[var_name].drop_vars("river", errors="ignore"))

        # Combine into a single DataArray with 'properties' as a new dimension
        river_da = xr.concat(river_arrays, dim="properties")
        river_list.append(river_da)

    # Concatenate all rivers along a new 'river_number' dimension
    rivers_combined = xr.concat(river_list, dim="river_number")
    return xr.Dataset({"rivers": rivers_combined}).assign_coords(properties=properties, river_number=river_values)


def add_river_coordinates(ds_rivers, df_rivers):
    # Extract geographical coordinates from df_rivers
    # Match river_number in ds_rivers with 'River number' in df_rivers
    lat_coords = []
    lon_coords = []
    for river_num in ds_rivers.river_number.values:
        # Convert river_num to int for matching
        river_row = df_rivers[df_rivers["River number"] == int(river_num)]
        if len(river_row) > 0:
            lat_coords.append(river_row["LatOutlet"].values[0])
            lon_coords.append(river_row["LonOutlet"].values[0])
        else:
            # If river not found, use NaN
            lat_coords.append(np.nan)
            lon_coords.append(np.nan)

    # Add coordinates to ds_rivers
    return ds_rivers.assign_coords(LatOutlet=("river_number", lat_coords), LonOutlet=("river_number", lon_coords))


def get_river_indices(ds, ds_rivers, surface_mask):
    river_indices = {}
    for river_num in ds_rivers.river_number.values:
        ds_river = ds_rivers.sel(river_number=river_num)
        lat, lon = ds_river.LatOutlet.values, ds_river.LonOutlet.values
        if lat <= ds.Ny.values.min() or lat >= ds.Ny.values.max():
            print(
                f"Processing river {(int(river_num),)} original location ({lat}, {lon}) validity: "
                f"{lat} is outside of {ds.Ny.values.min()}; {ds.Ny.values.max()}"
            )
            continue
        if lon <= ds.Nx.values.min() or lon >= ds.Nx.values.max():
            print(
                f"Processing river {(int(river_num),)} original location ({lat}, {lon}) validity: "
                f"{lon} is outside of {ds.Nx.values.min()}; {ds.Nx.values.max()}"
            )
            continue
        Ny_idx = np.argmin(np.abs(ds.Ny.values - lat))
        Nx_idx = np.argmin(np.abs(ds.Nx.values - lon))
        is_valid, message = is_valid_water_cell(surface_mask, Ny_idx, Nx_idx)
        print(f"Processing river {(int(river_num),)} original location ({lat}, {lon})validity: {is_valid},  original location ({Ny_idx}, {Nx_idx}), lat={lat:.4f}, lon={lon:.4f}, message: {message}")
        if not is_valid:
            Ny_valid, Nx_valid, distance = find_closest_valid_water_cell(surface_mask, Ny_idx, Nx_idx)
            if Ny_valid is not None and Nx_valid is not None:
                print(
                    f"River {(int(river_num),)} original location ({Ny_idx}, {Nx_idx}) is invalid. "
                    f"Closest valid water cell found at ({Ny_valid}, {Nx_valid}) with distance {distance:.2f}."
                )
                Ny_idx, Nx_idx = Ny_valid, Nx_valid
            else:
                print(
                    f"River {(int(river_num),)} original location ({Ny_idx}, {Nx_idx}) is invalid. "
                    f"No valid water cell found within search radius."
                )
        river_indices[int(river_num)] = (Ny_idx, Nx_idx)
    return river_indices


def is_valid_water_cell(mask, ny, nx):
    """Check if (ny, nx) is a valid water cell.

    Valid water cell: value is 1 and has at least one nearby cell with value 0
    """
    # Check bounds
    if ny < 0 or ny >= mask.shape[0] or nx < 0 or nx >= mask.shape[1]:
        return False, "Index out of bounds"

    # Check if current cell is 1 (water)
    if mask[ny, nx] != 1:
        return False, f"Cell value is {mask[ny, nx]}, expected 1 for water"

    # Check 8 neighboring cells for at least one with value 1
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue  # Skip the center cell
            ny_neighbor = ny + dy
            nx_neighbor = nx + dx
            if 0 <= ny_neighbor < mask.shape[0] and 0 <= nx_neighbor < mask.shape[1]:
                neighbors.append(mask[ny_neighbor, nx_neighbor])

    has_boundary = any(n == 0 for n in neighbors)

    if not has_boundary:
        return False, "No neighboring cell with value 0 (not at water boundary)"

    return True, f"Valid water cell at boundary original location ({ny}, {nx})"


def find_closest_valid_water_cell(mask, ny, nx, max_search_radius=10):
    """Find the closest valid water cell to (ny, nx).

    Args:
        mask: 2D numpy array with mask values
        ny: y-index (row) of the original cell
        nx: x-index (column) of the original cell
        max_search_radius: maximum search distance in grid cells

    Returns:
        tuple: (ny_valid, nx_valid, distance) if found, or (None, None, None) if no valid cell found
    """
    # Check if the current cell is already valid
    is_valid, _ = is_valid_water_cell(mask, ny, nx)
    if is_valid:
        return ny, nx, 0

    # Search in expanding circles
    for radius in range(1, max_search_radius + 1):
        # Generate all points at this radius (roughly circular search)
        candidates = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                # Only check points roughly at this radius
                distance = np.sqrt(dy**2 + dx**2)
                if radius - 0.5 <= distance <= radius + 0.5:
                    ny_candidate = ny + dy
                    nx_candidate = nx + dx

                    # Check if this candidate is valid
                    is_valid, _ = is_valid_water_cell(mask, ny_candidate, nx_candidate)
                    if is_valid:
                        candidates.append((ny_candidate, nx_candidate, distance))

        # If we found valid candidates at this radius, return the closest one
        if candidates:
            # Sort by actual distance and return the closest
            candidates.sort(key=lambda x: x[2])
            return candidates[0]

    # No valid cell found within max_search_radius
    return None, None, None
