import numpy as np
import xarray as xr

from pyparallax import futils as _futils


def tri_interp2d(src_x, src_y, src_v, src_priority, dst_x, dst_y, undef=-999.0, as_xarray=True):
    """
    Accelerated triangle interpolation.

    Parameters
    ----------
    src_x : numpy.ndarray
        2-dimensional array of x in source coordinate
    src_y : numpy.ndarray
        2-dimensional array of y in source coordinate
    src_v : numpy.ndarray
        interpolation target value at (`src_y`, `src_x`)
    src_priority : numpy.ndarray
        priority at (`src_y`, `src_x`), priority is used when triangle is overlapped
    dst_x : numpy.ndarray
        2-dimensional array of x in source coordinate
    dst_y : numpy.ndarray
        2-dimensional array of y in source coordinate
    undef : float, optional
        undefined value, default -999.0
    as_xarray : bool, optional
        If True, return values as xr.DataArray

    Returns
    -------
    dst_v : numpy.ndarray or xr.DataArray
        interpolated value at (`src_y`, `src_x`)
    dst_priority : numpy.ndarray or xr.DataArray
        interpolated priority at (`src_y`, `src_x`)

    Note
    ----
    `dst_x`, `dst_y` must be equally spaced
    """
    invalid_index = (src_v==np.nan) + (src_priority==np.nan)
    src_v = np.where(invalid_index, undef, src_v)
    src_priority = np.where(invalid_index, undef, src_priority)

    do_flip = dst_y[0] > dst_y[-1]
    if do_flip:
        dst_y = np.flip(dst_y)
    
    dst_v, dst_priority = _futils.tri_interp2d(src_x, src_y, src_v, src_priority, dst_x, dst_y, undef)
    undef_index = (dst_v == undef)
    dst_v[undef_index] = np.nan
    dst_priority[undef_index] = np.nan
    
    if do_flip:
        dst_y = np.flip(dst_y)
        dst_v = np.flipud(dst_v.T)
        dst_priority = np.flipud(dst_priority.T)
    else:
        dst_v = dst_v.T
        dst_priority = dst_priority.T
    
    if as_xarray:
        dst_v = xr.DataArray(dst_v, dims=("y","x"), coords={"y":dst_y, "x":dst_x})
        dst_priority = xr.DataArray(dst_priority, dims=("y","x"), coords={"y":dst_y, "x":dst_x})
    return dst_v, dst_priority