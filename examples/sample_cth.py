# %%
import numpy as np
import xarray as xr

import pyparallax


# Dummy data for testing
nx, ny, nz = 1, 1, 32
z = np.arange(nz) # Just an index
y = np.arange(ny)
x = np.arange(nx)

# 3D reference temperature (simple lapse rate + noise)
reftemp = xr.DataArray(
    300 - 6.5 * 16/nz * np.broadcast_to(z[:, None, None], (nz, ny, nx)) + np.random.randn(nz, ny, nx),
    dims=("z", "y", "x"),
    coords={"z": z, "y": y, "x": x}
)

# Create 3D heights matching reftemp shape
# Simple linear profile in z for all columns
heights = xr.DataArray(
    16/nz * np.broadcast_to(z[:, None, None], (nz, ny, nx)),
    dims=("z", "y", "x"),
    coords={"z": z, "y": y, "x": x}
)

# Observed temperature at ~10 km height with small x-dependent perturbation
temp_nx, temp_ny = 50, 50
temp_y = np.arange(temp_ny)
temp_x = np.arange(temp_nx)
mean_z = 10 # km

# 2D observed temperature
temp = xr.DataArray(
    300 - 6.5*mean_z + 13*np.sin(np.linspace(0, 4 * np.pi, temp_nx))[None, :].repeat(temp_ny, axis=0),
    dims=("y", "x"),
    coords={"y": temp_y, "x": temp_x}
)


# Interp reftemp and heights to temp grid (nearest neighbor in this example)
reftemp_interp = reftemp.interp(
    y=temp["y"].data, x=temp["x"].data, 
    method="nearest", kwargs={"fill_value": "extrapolate"}
)
heights_interp = heights.interp(
    y=temp["y"].data, x=temp["x"].data, 
    method="nearest", kwargs={"fill_value": "extrapolate"}
)


# Compute matched height
matched = pyparallax.utils.match_temperature_height(
    temp, reftemp_interp, heights_interp, prefer='lower'
)
#%% Plot result
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(7.5,5))
gs = GridSpec(
    2, 2, figure=fig,
    wspace=0.3, hspace=0.4, 
    width_ratios=[1,1],
    )
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0])
gsr = gs[:,1].subgridspec(2, 1, height_ratios=[1,0.1])
ax3 = fig.add_subplot(gsr[0])
ax = [ax1, ax2, ax3]

for i, iax in enumerate(ax):
    if i in (0,1):
        if i == 0:
            mp = iax.pcolormesh(temp.x, temp.y, temp, cmap="coolwarm", shading='auto')
            title = 'Observed temperature (K)'
        if i == 1:
            mp = iax.pcolormesh(matched.x, matched.y, matched, cmap="Spectral_r", shading='auto')
            title = 'Matched Height (km)'
        iax.set(xlabel='x', ylabel='y', aspect='equal')
        p = iax.get_position()
        cax = fig.add_axes([p.x1 + 0.05*p.width, p.y0, 0.04*p.width, p.height])
        cb = fig.colorbar(mp, cax=cax)
        cax.tick_params(direction='in')
    if i == 2:
        mean_reftemp = reftemp.mean(['y',"x"])
        mean_height = heights.mean(['y',"x"])
        iax.plot(mean_reftemp, mean_height, color='black', label='Mean ref temp')
        title = 'Reference temperature profile'
        
        iax.xaxis.set_major_locator(plt.MultipleLocator(20))
        iax.yaxis.set_major_locator(plt.MultipleLocator(1))

        iax.axvline(temp.mean(), color="C0", linestyle="--", label="Mean obs temp")
        iax.axvspan(
            temp.mean() - temp.std(), 
            temp.mean() + temp.std(), 
            color="C0", alpha=0.2, label="Obs temp ± 1σ"
        )
        iax.axhline(matched.mean(), color="C1", linestyle="--", label="Mean matched height")
        iax.axhspan(
            matched.mean() - matched.std(), 
            matched.mean() + matched.std(), 
            color="C1", alpha=0.2, label="Matched height ± 1σ"
        )
        iax.set(xlabel='Mean reference temperature (K)', ylabel='Height (km)', ylim=(0,16))
        iax.legend(
            frameon=False, loc='upper center', 
            bbox_to_anchor=(0.45, -0.13), ncol=2,
            fontsize='small', handlelength=1.2
        )    
    iax.set_title(title, loc='left')
    iax.tick_params(direction='in', top=True, right=True)

# %%
