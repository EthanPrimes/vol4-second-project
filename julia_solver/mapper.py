import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from scipy.interpolate import splprep, splev

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

# %matplotlib inline
# plt.rcParams['figure.dpi'] = 150
# plt.rcParams['font.family'] = 'sans-serif'


# ── Race marks: (lon, lat) — NOTE: cartopy wants lon first ──────────────────
MARKS = {
    "Start/Finish\n(Menominee)": (-87.614, 45.108),
    "Green Island":              (-87.495, 44.990),
    "Fish Creek Buoy":           (-87.245, 45.135),
    "Strawberry Ch. S":          (-87.260, 45.180),
    "Strawberry Ch. N":          (-87.270, 45.215),
    "Chambers Island":           (-87.375, 45.200),
}

# Official course sequence (lon, lat)
OFFICIAL = [
    (-87.614, 45.108),   # Start
    (-87.495, 44.990),   # Green Island
    (-87.245, 45.135),   # Fish Creek
    (-87.260, 45.180),   # Strawberry S
    (-87.270, 45.215),   # Strawberry N
    (-87.375, 45.200),   # Chambers Island
    (-87.614, 45.108),   # Finish
]

# ── Islands / obstacles: (lon, lat, label, radius_deg_approx) ───────────────
OBSTACLES = [
    (-87.357, 45.184, "", 0.035)
    # (-87.370, 45.162, "Horseshoe\nIsland",    0.012),
    # (-87.268, 45.196, "Strawberry\nIslands",  0.018),
    # (-87.495, 44.990, "Green Island",         0.010),
    # (-87.340, 45.215, "Hat Island",           0.008),
    # (-87.290, 45.175, "Nicolet Bay\nShoals",  0.015),
]

# ── Map extent: [lon_min, lon_max, lat_min, lat_max] ────────────────────────
EXTENT = [-87.80, -87.10, 44.88, 45.35]

def smooth_route(waypoints, n=400):
    """Fit a cubic spline through (lon, lat) waypoints."""
    lons = np.array([p[0] for p in waypoints])
    lats = np.array([p[1] for p in waypoints])
    k = min(3, len(waypoints) - 1)
    tck, _ = splprep([lons, lats], s=0, k=k)
    u = np.linspace(0, 1, n)
    lons_s, lats_s = splev(u, tck)
    return list(zip(lons_s, lats_s))


PLATE = ccrs.PlateCarree()


def add_basemap_features(ax):
    """Add land, water, and coastline features to a cartopy axis."""
    ax.add_feature(cfeature.OCEAN.with_scale('10m'),      facecolor='#c8ddf0', zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('10m'),       facecolor='#e8e4d9', zorder=1)
    ax.add_feature(cfeature.LAKES.with_scale('10m'),      facecolor='#c8ddf0', zorder=1)
    ax.add_feature(cfeature.RIVERS.with_scale('10m'),     edgecolor='#a0bcd0', linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'),  edgecolor='#6a8fa8', linewidth=0.8, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'),    edgecolor='#aaaaaa', linewidth=0.5, linestyle=':', zorder=3)
    ax.add_feature(cfeature.STATES.with_scale('10m'),     edgecolor='#bbbbbb', linewidth=0.4, linestyle='--', zorder=3)


def draw_course(ax, waypoints, color='#222222', lw=1.6,
                linestyle='--', label='Official course', zorder=5):
    lons = [p[0] for p in waypoints]
    lats = [p[1] for p in waypoints]
    ax.plot(lons, lats, linestyle=linestyle, color=color,
            linewidth=lw, transform=PLATE, zorder=zorder, label=label)


def draw_route(ax, route, color, label, lw=2.2, zorder=6):
    lons = [p[0] for p in route]
    lats = [p[1] for p in route]
    ax.plot(lons, lats, color=color, linewidth=lw,
            transform=PLATE, zorder=zorder, label=label, alpha=0.88)
    # Direction arrow at ~60% along the route
    mid = int(len(route) * 0.6)
    ax.annotate('', xy=(lons[mid+3], lats[mid+3]),
                xytext=(lons[mid], lats[mid]),
                xycoords=PLATE._as_mpl_transform(ax),
                textcoords=PLATE._as_mpl_transform(ax),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8),
                zorder=zorder+1)


def draw_marks(ax, marks, zorder=10):
    for i, (name, (lon, lat)) in enumerate(marks.items(), start=1):
        is_start = i == 1
        color = '#111111' if is_start else '#cc2200'
        marker = 's' if is_start else 'o'
        ax.plot(lon, lat, marker=marker, color=color, markersize=18 if is_start else 15,
                markeredgecolor='white', markeredgewidth=1.6,
                transform=PLATE, zorder=zorder)
        # Number badge
        ax.text(lon, lat, str(i), color='white', fontsize=9, fontweight='bold',
                ha='center', va='center', transform=PLATE, zorder=zorder+1)
        # Label — nudge to avoid overlap
        nudge = {1: (0.010, 0.006), 2: (-0.045, -0.018),
                 3: (0.010, 0.006), 4: (0.012, -0.005),
                 5: (0.012, 0.005), 6: (-0.055, 0.02)}
        dx, dy = nudge.get(i, (0.008, 0.006))
        ax.text(lon + dx, lat + dy, name,
                fontsize=11.25, fontweight='bold', color='white', transform=PLATE,
                zorder=zorder+1, ha='left')
                # path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])


def draw_obstacles(ax, obstacles, zorder=4):
    for lon, lat, label, r in obstacles:
        # Filled circle for island body
        circ = mpatches.Ellipse((lon, lat), width=r*1.4, height=r,
                                 facecolor='#c8b89a', edgecolor='#8a6a40',
                                 linewidth=1.0, transform=PLATE, zorder=zorder,
                                 alpha=0.85)
        ax.add_patch(circ)
        # Dashed exclusion ring
        ring = mpatches.Ellipse((lon, lat), width=(r+0.008)*1.4, height=r+0.008,
                                 facecolor='none', edgecolor='#e6550d',
                                 linewidth=0.8, linestyle=':', transform=PLATE,
                                 zorder=zorder+1, alpha=0.7)
        ax.add_patch(ring)
        ax.text(lon, lat - r - 0.006, label, fontsize=9,
                color='#7f2704', ha='center', transform=PLATE, zorder=zorder+2,
                path_effects=[pe.withStroke(linewidth=3, foreground='white')])


def add_legend(ax, extra_handles=None):
    handles = [
        Line2D([0],[0], color='#222', lw=1.6, linestyle='--', label='Official course'),
        mpatches.Patch(facecolor='#c8b89a', edgecolor='#8a6a40', label='Island / shoal'),
        mpatches.Patch(facecolor='none', edgecolor='#e6550d',
                       linestyle=':', label='Exclusion zone'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#cc2200',
               markeredgecolor='white', markersize=11, label='Race mark'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor='#111',
               markeredgecolor='white', markersize=11, label='Start / Finish'),
    ]
    if extra_handles:
        handles += extra_handles
    ax.legend(handles=handles, loc='lower right', fontsize=11,
              framealpha=0.92, edgecolor='#cccccc', fancybox=False)


def add_scalebar(ax, lon0, lat0, length_nm=10):
    """Rough scale bar — 1 nm ≈ 1/60 degree latitude."""
    deg = length_nm / 60.0
    ax.plot([lon0, lon0 + deg], [lat0, lat0], 'k-', lw=3, transform=PLATE, zorder=20)
    ax.text(lon0 + deg/2, lat0 + 0.004, f'{length_nm} nm',
            ha='center', fontsize=9, transform=PLATE, zorder=20)
    

def get_default_map():
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())

    ax.set_extent(EXTENT, crs=PLATE)
    add_basemap_features(ax)

    draw_obstacles(ax, OBSTACLES)
    draw_course(ax, OFFICIAL, label='Official course')
    draw_marks(ax, MARKS)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='#aaaaaa',
                    linestyle=':', x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}

    add_scalebar(ax, -87.76, 44.91, length_nm=10)
    add_legend(ax)

    ax.set_title('MMYC 100 Miler — Green Bay Course', fontsize=15, pad=10)
    fig.patch.set_facecolor('white')

    # plt.tight_layout()
    # plt.savefig('mmyc_100_miler_full.png', dpi=200, bbox_inches='tight')
    # plt.show()

    return ax

def get_satellite_map():
    # Stamen Terrain gives a nice topographic feel for coastal sailing maps
    tiler = cimgt.GoogleTiles(style='satellite')   # swap to 'terrain' or 'street' if preferred

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(1, 1, 1, projection=tiler.crs)

    ax.set_extent(EXTENT, crs=PLATE)
    ax.add_image(tiler, 11)   # zoom level 11 — increase for more detail, slower fetch

    draw_obstacles(ax, OBSTACLES)
    draw_course(ax, OFFICIAL, color='#222222', label='Official course')
    draw_marks(ax, MARKS)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='#aaaaaa',
                    linestyle=':', x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 9, 'color': '#111111'}
    gl.ylabel_style = {'size': 9, 'color': '#111111'}

    add_legend(ax)
    ax.set_title('MMYC 100 Miler — Satellite View', fontsize=15, pad=10, color='#111111')
    fig.patch.set_facecolor('white')

    # plt.tight_layout()
    # plt.savefig('mmyc_100_miler_satellite.png', dpi=200, bbox_inches='tight')
    # plt.show()
    return ax