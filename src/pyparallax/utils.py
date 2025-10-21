#%%
import numpy as np
import xarray as xr


def match_temperature_height(temp, reftemp, heights, prefer="lower", xname="x", yname="y", zname="z"):
    """
    Compute the altitude where reftemp matches 2D temp (y, x) using linear interpolation.
    Input heights is 3D (z, y, x) to allow non-uniform vertical grids.

    Parameters
    ----------
    temp : xr.DataArray, shape (y, x)
        2D observed brightness temperature.
    reftemp : xr.DataArray, shape (z, y, x)
        3D reference temperature profiles. `y` and `x` must match temp.
    heights : xr.DataArray or np.ndarray, shape (z, y, x)
        3D altitudes corresponding to each (z,y,x) point. 
        Coordinates must match reftemp.
    prefer : {'lower', 'upper'}, default='lower'
        Choose which intersection to use if multiple exist.
    xname: str, default='x'
        Name of x dimension.
    yname: str, default='y'
        Name of y dimension.
    zname: str, default='z'
        Name of vertical dimension.

    Returns
    -------
    matched_height : xr.DataArray, shape (y, x)
        Altitude where reftemp == temp after linear interpolation. 
        `np.nan` if no match.
    """
    # Coordinate checks
    if temp.coords[yname].shape != reftemp.coords[yname].shape or not np.all(temp.coords[yname] == reftemp.coords[yname]):
        raise ValueError("y coordinates of temp and reftemp do not match")
    if temp.coords[xname].shape != reftemp.coords[xname].shape or not np.all(temp.coords[xname] == reftemp.coords[xname]):
        raise ValueError("x coordinates of temp and reftemp do not match")
    if heights.shape != reftemp.shape:
        raise ValueError("heights shape must match reftemp shape")
        
    
    # Convert to DataArray if necessary
    if not isinstance(heights, xr.DataArray):
        heights = xr.DataArray(heights, dims=reftemp.dims, coords=reftemp.coords)

    # Lower and upper layers
    t0 = reftemp.isel({zname: slice(None, -1)})
    t1 = reftemp.isel({zname: slice(1, None)})
    z0 = heights.isel({zname: slice(None, -1)})
    z1 = heights.isel({zname: slice(1, None)})

    # Temperature difference
    diff0 = t0 - temp
    diff1 = (t1 - temp).assign_coords({zname: diff0.z})

    # Detect intersections
    sign_change = (diff0 * diff1 < 0)

    # Linear interpolation factor
    with np.errstate(invalid='ignore', divide='ignore'):
        frac = diff0 / (diff0 - diff1)

    # Interpolated height
    z_cross = z0 + (z1 - z0) * frac

    # Mask invalid intersections
    z_cross = z_cross.where(sign_change)

    # Choose lower or upper intersection
    if prefer == 'lower':
        matched_height = z_cross.min(dim=zname, skipna=True)
    elif prefer == 'upper':
        matched_height = z_cross.max(dim=zname, skipna=True)
    else:
        raise ValueError("prefer must be 'lower' or 'upper'")

    # Add metadata
    matched_height.name = "matched_height"
    matched_height.attrs['description'] = "Altitude where reftemp == temp (linear interpolation)"

    return matched_height