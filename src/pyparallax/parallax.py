import pyproj
import numpy as np

from pyparallax import interp


def correction(src_lon, src_lat, src_val, cth, satlon, satlat=0.0, satheight=42164.0, proj=None, dst_x=None, dst_y=None, dst_is_lonlat=True, undef=-99999999.0, return_latlon_corr=False):
    """
    Performs parallax correction to variables observed from the satellite.
    Forward mapping is done on the map projection surface (i.e., x-y plane).

    Parameters
    ----------
    src_lon: np.ndarray of float
        Longitude of each grid point
    src_lat: np.ndarray of float
        Latitude of each grid point
    src_val: np.ndarray of float
        The variables to be corrected.
    cth : np.ndarray of float
        The cloud top height for each grid [km].
        `cth` must have the same size and projection as the `values`.
    satlon: float
        Longitude of the satellite (degrees east).
    satlat: float, default 0
        Latitude of the satellite (degrees north).
    satheight: float, default 42164.0
        Height of the satellite measured from center of the Earth (km).
    proj: pyproj.Proj, optional
        Projection of the corrected values.
        If not specified, `proj` will be Geostationary Satellite View using WGS84 ellipses.
    dst_x: np.ndarray of float, optional
        x-coordinate of each destination grid point. `dst_x` must be equally spaced 1d array.
        If not specified, the destination is set to the same position as the input.
    dst_y: np.ndarray of float, optional
        y-coordinate of each destination grid point. `dst_y` must be equally spaced 1d array.
        If not specified, the destination is set to the same position as the input.
    dst_is_lonlat: bool, default True
        If True, `dst_x` and `dst_y` are interpreted as longitude and latitude.
        If False, `dst_x` and `dst_y` are interpreted as `x` and `y` in the projection. 
    undef: float
        Undefined value used in Fortran module.
    """
    if src_lon.shape != src_lat.shape != src_val.shape != cth.shape:
        raise ValueError("`src_lon`, `src_lat`, `src_val`, and `cth` must have same shape")
    if dst_x is not None and dst_x.ndim != 1:
        raise ValueError("dst_x must be 1-d array")
    if dst_y is not None and dst_y.ndim != 1:
        raise ValueError("dst_y must be 1-d array")

    if proj is None:
        wgs84 = pyproj.CRS("EPSG:4326")
        h = satheight*1e3 - wgs84.get_geod().a
        proj = pyproj.Proj(f"+proj=geos +h={h} +lon_0={satlon} +lat_0={satlat} +sweep=x")
    r_a = proj.crs.get_geod().a
    r_b = proj.crs.get_geod().b

    # shifting pixels according to parallax corretion
    lat_corr, lon_corr = calc_parallax_shift(
        cth=cth, lat=src_lat, lon=src_lon, satheight=satheight,
        satlat=satlat, satlon=satlon, radius_eq=r_a*1e-3, radius_pole=r_b*1e-3)

    if return_latlon_corr:
        return lon_corr, lat_corr
    x_corr, y_corr = proj(lon_corr, lat_corr)

    if dst_x is None or dst_y is None:
        dst_xx, dst_yy = proj(src_lon, src_lat)
        dst_x = dst_xx[dst_xx.shape[0]//2,:]
        dst_y = dst_yy[:,dst_xx.shape[1]//2]
    else:
        if dst_is_lonlat is True:
            dst_xx, dst_yy = np.meshgrid(dst_x, dst_y)
            dst_xx, dst_yy = proj(dst_xx, dst_yy)
            dst_x = dst_xx[dst_xx.shape[0]//2,:]
            dst_y = dst_yy[:,dst_xx.shape[1]//2]

    ovalues, octh = perform_correction(x_corr, y_corr, src_val, cth, dst_x, dst_y, undef)
    return ovalues, octh

def perform_correction(x_corr, y_corr, src_val, cth, dst_x, dst_y, undef=-99999999.0, as_xarray=True):
    """
    Performs parallax correction to variables observed from the satellite.
    Forward mapping is done on the map projection surface (i.e., x-y plane).

    Parameters
    ----------
    """
    if dst_x is not None and dst_x.ndim != 1:
        raise ValueError("dst_x must be 1-d array")
    if dst_y is not None and dst_y.ndim != 1:
        raise ValueError("dst_y must be 1-d array")
    
    ovalues, octh = interp.tri_interp2d(x_corr, y_corr, src_val, cth, dst_x, dst_y, undef, as_xarray=as_xarray)
    return ovalues, octh

def calc_parallax_shift(cth, lat, lon, satheight=42164.0, satlat=0.0, satlon=140.7, radius_eq=6378.1370, radius_pole=6356.7523, ellps=None):
    """
    Calcurate parallax of clouds observed by satellites.

    Parameters
    ----------
    cth : float or np.ndarray
        Cloud top height for each grid point [km].
    lat : float or numpy.ndarray
        Latitude of cloud [degree]
    lon : float or numpy.ndarray
        Longitude of cloud [degree]
    satheight : float, default 42164.0
        Height of satellite from Earth center [km], default: 42164.0 (for Himawari-8)
    satlat : float, default 0.0
        Latitude of satellite [degree], default: 0.0 (for Himawari-8)
    satlon : float, default 140.7
        Longitude of satellite [degree], default: 140.7 (for Himawari-8)
    radius_eq : float, default 6378.1370
        Radius of semi-major axis, default: WGS84's length 6378.1370
    radius_pole : float, default 6356.7523
        Radius of semi-minor axis, default: WGS84's length 6356.7523
    ellps : str
        Geodetic parameters for specifying the ellipsoid

    Returns
    -------
    lat_corr : float
        Parallax corrected latitude
    lon_corr : float
        Parallax corrected longitude

    Notes
    -----
    If both 'ellps' and 'radius_eq(pole)' are specified, 'ellps' has priority.
    """
    # Argument checks
    if ellps is not None:
        geod = pyproj.Geod(ellps=ellps)
        radius_eq = geod.a*1e-3
        radius_pole = geod.b*1e-3

    if type(cth) in (float, int):
        cth = [cth]
    if type(lon) in (float, int):
        lon = [lon]
    if type(lat) in (float, int):
        lat = [lat]
    
    cth = np.array(cth).astype(np.float64)
    lon = np.array(lon).astype(np.float64)
    lat = np.array(lat).astype(np.float64)

    if cth.shape != lat.shape or cth.shape != lon.shape:
        raise ValueError("cth and lat and lon must have same shape")

    # degrees to radians
    satlat = np.deg2rad(satlat)
    satlon = np.deg2rad(satlon)
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)

    # Cartesian coordinates for the satellite
    xsat = satheight * np.cos(satlat) * np.sin(satlon)
    ysat = satheight * np.sin(satlat)
    zsat = satheight * np.cos(satlat) * np.cos(satlon)

    # Cartesian coordinates of the surface point
    radius_ratio = radius_eq/radius_pole
    radius_local = radius_eq/np.sqrt(np.cos(lat)**2 + radius_ratio**2 * np.sin(lat)**2)
    xcloud = radius_local * np.cos(lat) * np.sin(lon)
    ycloud = radius_local * np.sin(lat)
    zcloud = radius_local * np.cos(lat) * np.cos(lon)

    # Calcurate the diff vector
    xdiff = xsat - xcloud
    ydiff = ysat - ycloud
    zdiff = zsat - zcloud

    # Compute new radius ratio depending on cth
    radius_ratio_local = ((radius_eq+cth)/(radius_pole+cth))**2

    # Equation to solve for the line of sight at cth
    e1 = xdiff**2 + radius_ratio_local*ydiff**2 + zdiff**2
    e2 = 2.0 * (xcloud*xdiff + radius_ratio_local*ycloud*ydiff + zcloud*zdiff)
    e3 = xcloud**2 + zcloud**2 + radius_ratio_local*ycloud**2 - (radius_eq+cth)**2
    c = (np.sqrt(e2**2 - 4.0*e1*e3) - e2)/2.0/e1

    # Calcurate Corrected cloud-top coordinates
    xcorr = xcloud + c * xdiff
    ycorr = ycloud + c * ydiff
    zcorr = zcloud + c * zdiff

    # Convert back from x,y,z to lon,lat
    tangent_lat_geod_corr = radius_ratio_local * ycorr/np.hypot(xcorr, zcorr)
    lat_cloud_corr = np.arctan(tangent_lat_geod_corr/(radius_ratio**2))
    lat_corr = np.rad2deg(lat_cloud_corr)

    eps = np.abs(zcorr/xcorr)
    r_A = radius_eq/np.sqrt(np.cos(lat_cloud_corr)**2 + (radius_ratio**2) * (np.sin(lat_cloud_corr)**2))
    x_A = np.sign(xcorr) * r_A * np.cos(lat_cloud_corr) / np.sqrt(1.0+eps**2)
    z_A = np.sign(zcorr) * eps * np.abs(x_A)
    lon_corr = np.rad2deg(np.arctan2(x_A, z_A))
    on_nadir_lon = (xcorr==0)
    lon_corr[on_nadir_lon] = lon[on_nadir_lon]
    return lat_corr, lon_corr