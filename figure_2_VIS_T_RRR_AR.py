import os
import re
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer

# =========================================================
# === 1. CONFIGURACIÓN BASE Y FUNCIONES DE PLOTEO
# =========================================================
def setup_map(ax, proj_map, proj_data):
    """Configura el mapa base en un eje Cartopy."""
    ax.coastlines(resolution='110m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

def plot_temp_z500(ax, lon, lat, temp_field, z500_field,
                   levels_temp, levels_z500, proj_data):
    # --- Temperature filled contours ---
    temp_plot = ax.contourf(
        lon, lat, temp_field,
        levels=levels_temp,
        cmap='bwr',
        extend='both',
        transform=proj_data
    )
    # --- Z500 gray contour lines ---
    c = ax.contour(
        lon, lat, z500_field,
        levels=levels_z500,
        colors='gray',
        linewidths=1.0,
        transform=proj_data
    )
    ax.clabel(c, fmt='%d', fontsize=7, colors='gray')
    return temp_plot

def plot_precip(ax, lon, lat, precip_field, levels_precip, cmap_precip, proj_data):
    precip_plot = ax.contourf(
        lon, lat, precip_field,
        levels=levels_precip,
        cmap=cmap_precip,
        extend='max',
        transform=proj_data
    )
    return precip_plot

def plot_shapefile_mask(ax, lon, lat, mask, color='magenta', lw=2, proj_data=ccrs.PlateCarree()):
    ax.contour(lon, lat, mask, levels=[0.5], colors=color,
               linewidths=lw, transform=proj_data)

def normalize(band):
    band_min, band_max = np.nanpercentile(band, (2, 98))
    return np.clip((band - band_min) / (band_max - band_min), 0, 1)

def read_rgb_tiff_crop(tif_path, crop_extent):
    with rasterio.open(tif_path) as src:
        crs = src.crs
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        left, bottom = transformer.transform(crop_extent[0], crop_extent[2])
        right, top = transformer.transform(crop_extent[1], crop_extent[3])
        window = from_bounds(left, bottom, right, top, src.transform)

        r = src.read(1, window=window).astype(float)
        g = src.read(2, window=window).astype(float)
        b = src.read(3, window=window).astype(float)
        lon_min, lat_min = crop_extent[0], crop_extent[2]
        lon_max, lat_max = crop_extent[1], crop_extent[3]

    r_norm, g_norm, b_norm = normalize(r), normalize(g), normalize(b)
    rgb = np.dstack((r_norm, g_norm, b_norm))
    extent = [lon_min, lon_max, lat_min, lat_max]
    return rgb, extent

# =========================================================
# === 2. CARGA DE DATOS
# =========================================================
output_dir = "./fig"
os.makedirs(output_dir, exist_ok=True)

# Datasets ERA5 y AR
t2m = xr.open_dataset('~/ERA5/T2max_daily_anomalies.nc')
rrr = xr.open_dataset('~/ERA5/precip_daily_totals.nc')
z500 = (xr.open_dataset('~/ERA5/Z500_daily_anomalies_75.nc') / 9.98).squeeze()

AR_shape = xr.open_dataset('/home/cr2/cmtorres/ARs_DATA/Global_AR_dataset_Guan.nc').sel(
    time=slice('1980-01-01', '2022-12-31'), ens=1, lev=0
)
AR_shape.coords['lon'] = (AR_shape.coords['lon'] + 180) % 360 - 180
AR_shape = AR_shape.sortby(AR_shape.lon)

try:
    AR_shape_B = xr.open_dataset('/home/cr2/cmtorres/ARs_DATA/Global_AR_dataset_B.nc')
except Exception:
    AR_shape_B = None

# Colormap para precipitación
colors = [
    "#f0f0f0", "#7ea6ff", "#3366ff", "#1e90ff", "#00c5cd",
    "#00c878", "#32cd32", "#9acd32", "#ffd54f", "#ffb347",
    "#ff7f50", "#ff4500", "#ff1493", "#c050ff", "#6a3d9a"
]
cmap_precip = ListedColormap(colors)

# =========================================================
# === 3. CONFIGURACIÓN DE CASOS (AÑO, FECHA, TIF)
# =========================================================
cases = [
    {
        "label": "(a) 2019",
        "date": "2019-02-04",
        "tif": "/home/cr2/cmtorres/DATA/img_sat_lar/crop_2019-02-04.tif"
    },
    {
        "label": "(b) 2020",
        "date": "2020-02-10",
        "tif": "/home/cr2/cmtorres/DATA/img_sat_lar/crop_2020-02-10.tif"
    },
    {
        "label": "(c) 2021",
        "date": "2021-02-10",
        "tif": "/home/cr2/cmtorres/DATA/img_sat_lar/crop_2021-02-10.tif",
        "has_ar2": True
    },
    {
        "label": "(d) 2022",
        "date": "2022-02-07",
        "tif": "/home/cr2/cmtorres/DATA/img_sat_lar/crop_2022-02-07.tif"
    }
]

levels_temp = np.arange(-12, 13, 1)
levels_z500 = np.delete(np.arange(-300, 301, 50), 6)
levels_precip = np.arange(0, 15, 1.0)

proj_map = ccrs.Orthographic(-70, -60)
proj_data = ccrs.PlateCarree()
crop_extent = [-180, 180, -90, 10]

# =========================================================
# === 4. CREACIÓN DE LA FIGURA 4x3
# =========================================================
fig, axes = plt.subplots(
    4, 3,
    figsize=(12, 14),
    subplot_kw={"projection": proj_map},
    gridspec_kw={"wspace": 0.05, "hspace": 0.25}
)

temp_plot = None
precip_plot = None

for i, case in enumerate(cases):
    date_str = case["date"]
    panel_label = case["label"]
    
    # Extraer datos subconjunto
    t2m_sub = t2m.sel(time=date_str).tmax_anomaly
    z500_sub = z500.sel(time=date_str).z_anomaly
    rrr_sub = rrr.sel(time=date_str).precip_daily
    ar1_sub = AR_shape.sel(time=date_str).shapemap
    ar1_sub_12h = ar1_sub.sel(time=(ar1_sub.time.dt.hour == 12))
    
    lon, lat = t2m_sub.longitude, t2m_sub.latitude

    # --- Columna 0: VIS Satellite Image ---
    ax_vis = axes[i, 0]
    setup_map(ax_vis, proj_map, proj_data)
    try:
        rgb, extent2 = read_rgb_tiff_crop(case["tif"], crop_extent)
        ax_vis.imshow(rgb, origin="upper", extent=extent2, transform=proj_data)
    except Exception as e:
        print(f"No se pudo cargar TIFF para {date_str}: {e}")
    
    # Fecha centrada y etiqueta de año a la izquierda
    ax_vis.set_title(date_str, fontsize=10, loc='center')

    ax_vis.text(-0.08, 0.98, panel_label, transform=ax_vis.transAxes, 
                fontsize=13, fontweight='bold', ha='left', va='top')

    # --- Columna 1: Temperature & Z500 ---
    ax_temp = axes[i, 1]
    setup_map(ax_temp, proj_map, proj_data)
    temp_plot = plot_temp_z500(
        ax_temp, lon, lat, t2m_sub, z500_sub,
        levels_temp, levels_z500, proj_data
    )
    ax_temp.set_title(date_str, fontsize=10, loc='center')

    # --- Columna 2: Precipitation & AR Shapes ---
    ax_precip = axes[i, 2]
    setup_map(ax_precip, proj_map, proj_data)
    precip_plot = plot_precip(
        ax_precip, lon, lat, rrr_sub,
        levels_precip, cmap_precip, proj_data
    )
    ax_precip.set_title(date_str, fontsize=10, loc='center')

    # Máscaras AR
    if not ar1_sub_12h.isnull().all():
        ar1_step = ar1_sub_12h.isel(time=0)
        mask1 = ar1_step.isnull().astype(int)
        plot_shapefile_mask(ax_precip, ar1_step.lon, ar1_step.lat, mask1, color="magenta", lw=1.0)

    if case.get("has_ar2", False) and AR_shape_B is not None:
        try:
            ar2_sub = AR_shape_B.sel(time=date_str).ar_binary_tag
            ar2_sub_12h = ar2_sub.sel(time=(ar2_sub.time.dt.hour == 12))
            ar2_step = ar2_sub_12h.isel(time=0)
            plot_shapefile_mask(ax_precip, ar2_step.lon, ar2_step.lat, ar2_step, color="lime", lw=1.0)
        except Exception:
            pass

# =========================================================
# === 5. COLORBARS Y GUARDADO
# =========================================================
# Colorbar para Temperatura
cbar_ax1 = fig.add_axes([0.38, 0.08, 0.24, 0.015])
cbar1 = fig.colorbar(temp_plot, cax=cbar_ax1, orientation="horizontal")
cbar1.set_label("Air temperature anomaly (°C)", fontsize=10)

# Colorbar para Precipitación
cbar_ax2 = fig.add_axes([0.64, 0.08, 0.24, 0.015])
cbar2 = fig.colorbar(precip_plot, cax=cbar_ax2, orientation="horizontal")
cbar2.set_label("Total precipitation (mmWE day-1)", fontsize=10)

save_path = os.path.join(output_dir, "multipanel_2019_2022_labeled.png")
print(f"Guardando figura combinada en: {save_path}")
plt.savefig(save_path, dpi=200, bbox_inches="tight")
#plt.show()
