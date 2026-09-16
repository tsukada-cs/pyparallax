"""Validate calc_parallax_shift against an independent ray/ellipsoid intersection.

The reference solution intersects the line of sight with the surface of constant
geodetic height H, solved as a quadratic against the ellipsoid whose semi-axes
are (a+H, b+H). That surface differs from the true constant-height surface by
about H*f**2 (~1 m for H = 87 km), which is far below the tolerances used here.
"""

import numpy as np
import pyproj
import pytest

import pyparallax


A = 6378137.0
B = 6356752.314245

_TO_ECEF = pyproj.Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
_TO_GEO = pyproj.Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)


def reference_shift(cth_km, lat, lon, sat_ecef):
    """Exact parallax-corrected geodetic lat/lon for a target at cth_km."""
    h = cth_km * 1e3
    px, py, pz = _TO_ECEF.transform(lon, lat, np.zeros_like(lat))
    sx, sy, sz = sat_ecef
    dx, dy, dz = px - sx, py - sy, pz - sz

    a2, b2 = (A + h) ** 2, (B + h) ** 2
    qa = (dx**2 + dy**2) / a2 + dz**2 / b2
    qb = 2.0 * ((sx * dx + sy * dy) / a2 + sz * dz / b2)
    qc = (sx**2 + sy**2) / a2 + sz**2 / b2 - 1.0
    sq = np.sqrt(np.maximum(qb**2 - 4.0 * qa * qc, 0.0))
    t1, t2 = (-qb - sq) / (2.0 * qa), (-qb + sq) / (2.0 * qa)
    t = np.where((t1 >= 0.0) & (t1 <= 1.0), t1, t2)
    lon_c, lat_c, _ = _TO_GEO.transform(sx + t * dx, sy + t * dy, sz + t * dz)
    return lat_c, lon_c


def geodetic_sat_ecef(satlat_geocentric, satlon, satheight_km):
    """ECEF position of a satellite given geocentric lat/lon and radius."""
    r = satheight_km * 1e3
    la, lo = np.deg2rad(satlat_geocentric), np.deg2rad(satlon)
    return r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo), r * np.sin(la)


def separation_km(lat1, lon1, lat2, lon2):
    geod = pyproj.Geod(ellps="WGS84")
    _, _, d = geod.inv(lon1, lat1, lon2, lat2)
    return d / 1e3


def test_latitude_conversion_roundtrip():
    lat = np.linspace(-89.0, 89.0, 179)
    back = pyparallax.geocentric_to_geodetic_lat(
        pyparallax.geodetic_to_geocentric_lat(lat))
    np.testing.assert_allclose(back, lat, atol=1e-9)
    # the two conventions differ most near 45 deg
    diff = np.abs(lat - pyparallax.geodetic_to_geocentric_lat(lat))
    assert diff.max() == pytest.approx(0.1924, abs=1e-3)
    assert diff[np.argmin(np.abs(lat - 45.0))] == pytest.approx(0.1924, abs=1e-3)


def test_no_shift_at_sub_satellite_point():
    """Looking straight down, the target stays at the same lat/lon."""
    satlon, satlat = 140.7, 0.0
    lat = np.array([[0.0]])
    lon = np.array([[satlon]])
    lat_c, lon_c = pyparallax.calc_parallax_shift(
        cth=np.full_like(lat, 10.0), lat=lat, lon=lon,
        satlat=satlat, satlon=satlon, ellps="WGS84")
    assert separation_km(lat[0, 0], lon[0, 0], lat_c[0, 0], lon_c[0, 0]) < 1e-6


@pytest.mark.parametrize("cth_km", [5.0, 16.0, 87.0])
def test_matches_reference_geostationary(cth_km):
    """Geodetic convention must reproduce the exact ray/ellipsoid solution."""
    satlon, satlat, satheight = 140.7, 0.0, 42164.0
    lon2d, lat2d = np.meshgrid(np.arange(110.0, 175.0, 2.5),
                               np.arange(-55.0, 56.0, 2.5))
    cth = np.full_like(lat2d, cth_km)

    lat_c, lon_c = pyparallax.calc_parallax_shift(
        cth=cth, lat=lat2d, lon=lon2d, satheight=satheight,
        satlat=satlat, satlon=satlon, ellps="WGS84")
    ref_lat, ref_lon = reference_shift(
        cth_km, lat2d, lon2d, geodetic_sat_ecef(satlat, satlon, satheight))

    err = separation_km(lat_c, lon_c, ref_lat, ref_lon)
    assert np.nanmax(err) < 0.01, f"max error {np.nanmax(err):.4f} km"


@pytest.mark.parametrize("lat_deg,geo_err_km,leo_err_km", [
    (11.0, 0.014, 1.21),
    (30.0, 0.228, 2.35),
    (45.0, 0.820, 2.55),
    (60.0, 2.403, 2.07),
])
def test_geocentric_option_reproduces_previous_behaviour(lat_deg, geo_err_km, leo_err_km):
    """lat_is_geodetic=False keeps the old convention, which is off by this much.

    The error grows with latitude for a geostationary satellite, and is of order
    1-2.5 km at every latitude once the satellite itself is off the equator
    (i.e. for a cross-track scanner on low Earth orbit).
    """
    lat = np.array([[lat_deg]])
    lon = np.array([[150.0]])
    cth = np.full_like(lat, 87.0)

    geo = dict(satheight=42164.0, satlat=0.0, satlon=140.7)
    leo = dict(satheight=np.full_like(lat, 6371.0 + 834.0),
               satlat=lat + 3.0, satlon=np.full_like(lat, 150.0))

    for kw, expected in ((geo, geo_err_km), (leo, leo_err_km)):
        sat = geodetic_sat_ecef(kw["satlat"], kw["satlon"], kw["satheight"])
        ref_lat, ref_lon = reference_shift(87.0, lat, lon, sat)

        gd = pyparallax.calc_parallax_shift(cth=cth, lat=lat, lon=lon,
                                            ellps="WGS84", lat_is_geodetic=True, **kw)
        gc = pyparallax.calc_parallax_shift(cth=cth, lat=lat, lon=lon,
                                            ellps="WGS84", lat_is_geodetic=False, **kw)

        assert separation_km(gd[0], gd[1], ref_lat, ref_lon)[0, 0] < 0.01
        assert (separation_km(gc[0], gc[1], ref_lat, ref_lon)[0, 0]
                == pytest.approx(expected, rel=0.15))


def test_accepts_per_pixel_satellite_position():
    """Cross-track scanners need a different satellite position per scan line."""
    n = 7
    lat2d = np.tile(np.linspace(5.0, 20.0, n)[:, None], (1, 3))
    lon2d = np.tile(np.linspace(140.0, 150.0, 3)[None, :], (n, 1))
    satlat = np.tile(np.linspace(8.0, 22.0, n)[:, None], (1, 3))
    satlon = np.full_like(lat2d, 145.0)
    satheight = np.full_like(lat2d, 6371.0 + 834.0)

    lat_c, lon_c = pyparallax.calc_parallax_shift(
        cth=np.full_like(lat2d, 87.0), lat=lat2d, lon=lon2d,
        satheight=satheight, satlat=satlat, satlon=satlon, ellps="WGS84")

    sat = geodetic_sat_ecef(satlat, satlon, satheight)
    ref_lat, ref_lon = reference_shift(87.0, lat2d, lon2d, sat)
    assert np.nanmax(separation_km(lat_c, lon_c, ref_lat, ref_lon)) < 0.01
