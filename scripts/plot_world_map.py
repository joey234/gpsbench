#!/usr/bin/env python3
"""
Generate world map visualization showing geographic coverage of GPSBench.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from collections import defaultdict

# Try to import cartopy, fall back to basic plotting if not available
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Cartopy not installed. Using basic matplotlib plotting.")


def load_cities_from_tasks():
    """Extract all unique cities from task files."""
    cities = []
    seen = set()

    data_dir = Path(__file__).parent.parent / "data"

    # Load from both tracks
    for track in ["track_pure_gps", "track_applied"]:
        track_dir = data_dir / track
        if not track_dir.exists():
            continue

        for task_file in track_dir.glob("task*.json"):
            if "splits" in str(task_file):
                continue
            try:
                with open(task_file, 'r') as f:
                    samples = json.load(f)

                for sample in samples[:500]:  # Sample subset for visualization
                    # Extract coordinates from various task formats
                    coords = extract_coords(sample)
                    for lat, lon, name in coords:
                        key = f"{lat:.4f},{lon:.4f}"
                        if key not in seen:
                            seen.add(key)
                            cities.append({
                                'lat': lat,
                                'lon': lon,
                                'name': name,
                                'track': track
                            })
            except Exception as e:
                print(f"Error loading {task_file}: {e}")

    return cities


def extract_coords(sample):
    """Extract coordinates from a sample based on task type."""
    coords = []

    # Distance calculation, bearing, interpolation
    if 'point_a' in sample:
        coords.append((sample['point_a']['lat'], sample['point_a']['lon'],
                       sample['point_a'].get('name', '')))
    if 'point_b' in sample:
        coords.append((sample['point_b']['lat'], sample['point_b']['lon'],
                       sample['point_b'].get('name', '')))

    # Area/perimeter polygons
    if 'vertices' in sample:
        for v in sample['vertices']:
            coords.append((v['lat'], v['lon'], v.get('name', '')))

    # Bounding box
    if 'points' in sample and isinstance(sample['points'], list):
        for p in sample['points']:
            if isinstance(p, dict) and 'lat' in p:
                coords.append((p['lat'], p['lon'], p.get('name', '')))

    # Place association
    if 'coordinate' in sample and isinstance(sample['coordinate'], dict):
        c = sample['coordinate']
        if 'lat' in c:
            coords.append((c['lat'], c['lon'], ''))

    # Proximity
    if 'reference_point' in sample:
        rp = sample['reference_point']
        coords.append((rp['lat'], rp['lon'], rp.get('name', '')))
    if 'candidate_points' in sample:
        for cp in sample['candidate_points']:
            coords.append((cp['lat'], cp['lon'], cp.get('name', '')))

    # Route analysis
    if 'start' in sample and isinstance(sample['start'], dict):
        coords.append((sample['start']['lat'], sample['start']['lon'],
                       sample['start'].get('name', '')))
    if 'end' in sample and isinstance(sample['end'], dict):
        coords.append((sample['end']['lat'], sample['end']['lon'],
                       sample['end'].get('name', '')))
    if 'waypoint' in sample and isinstance(sample['waypoint'], dict):
        coords.append((sample['waypoint']['lat'], sample['waypoint']['lon'],
                       sample['waypoint'].get('name', '')))

    # Spatial patterns
    if 'points' in sample and sample.get('task') == 'spatial_patterns':
        for p in sample['points']:
            coords.append((p['lat'], p['lon'], p.get('name', '')))

    return coords


def get_continent(lat, lon):
    """Rough continent classification based on coordinates."""
    # Europe: includes Mediterranean
    if lat > 35 and -10 < lon < 60:
        return 'Europe'
    # Asia: includes Middle East, Central Asia, East Asia
    elif lat > -10 and 25 < lon < 150:
        return 'Asia'
    # Oceania: Australia, NZ, Pacific Islands
    elif (lat < 0 and 100 < lon < 180) or (lon > 150) or (-10 < lat < 20 and lon > 100):
        return 'Oceania'
    # Africa: includes all of Africa
    elif -35 < lat < 37 and -20 < lon < 55:
        return 'Africa'
    # South America: extended longitude range
    elif lat < 15 and -82 < lon < -30:
        return 'South America'
    # North America: includes Central America, Caribbean
    elif lat > -5 and -170 < lon < -30:
        return 'North America'
    else:
        return 'Other'


def plot_world_map_cartopy(cities, output_path):
    """Plot world map using cartopy."""
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

    # Add map features
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', edgecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='#e6f3ff')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#999999')
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='#cccccc', linestyle=':')

    # Color by continent
    continent_colors = {
        'Africa': '#e74c3c',
        'Asia': '#3498db',
        'Europe': '#2ecc71',
        'North America': '#9b59b6',
        'South America': '#f39c12',
        'Oceania': '#1abc9c',
        'Other': '#95a5a6'
    }

    # Group cities by continent
    continent_cities = defaultdict(list)
    for city in cities:
        continent = get_continent(city['lat'], city['lon'])
        continent_cities[continent].append(city)

    # Plot each continent
    for continent, cont_cities in continent_cities.items():
        lats = [c['lat'] for c in cont_cities]
        lons = [c['lon'] for c in cont_cities]
        ax.scatter(lons, lats,
                   c=continent_colors.get(continent, '#95a5a6'),
                   s=8, alpha=0.6,
                   transform=ccrs.PlateCarree(),
                   label=f'{continent} ({len(cont_cities):,})',
                   edgecolors='none')

    # Legend
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9,
              title='Cities by Continent', title_fontsize=10)

    # Title
    ax.set_title('GPSBench Geographic Coverage\n{:,} Unique Locations from GeoNames'.format(len(cities)),
                 fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_world_map_basic(cities, output_path):
    """Plot world map using basic matplotlib (fallback)."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color by continent
    continent_colors = {
        'Africa': '#e74c3c',
        'Asia': '#3498db',
        'Europe': '#2ecc71',
        'North America': '#9b59b6',
        'South America': '#f39c12',
        'Oceania': '#1abc9c',
        'Other': '#95a5a6'
    }

    # Background
    ax.set_facecolor('#e6f3ff')

    # Group cities by continent
    continent_cities = defaultdict(list)
    for city in cities:
        continent = get_continent(city['lat'], city['lon'])
        continent_cities[continent].append(city)

    # Plot each continent
    for continent, cont_cities in continent_cities.items():
        lats = [c['lat'] for c in cont_cities]
        lons = [c['lon'] for c in cont_cities]
        ax.scatter(lons, lats,
                   c=continent_colors.get(continent, '#95a5a6'),
                   s=10, alpha=0.6,
                   label=f'{continent} ({len(cont_cities):,})',
                   edgecolors='none')

    # Formatting
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title('GPSBench Geographic Coverage\n{:,} Unique Locations from GeoNames'.format(len(cities)),
                 fontsize=14, fontweight='bold')

    # Grid
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.axhline(y=0, color='#666', linewidth=0.5)
    ax.axvline(x=0, color='#666', linewidth=0.5)

    # Legend
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9,
              title='Cities by Continent', title_fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    # Load cities
    print("Loading cities from task files...")
    cities = load_cities_from_tasks()
    print(f"Found {len(cities):,} unique locations")

    # Print continent distribution
    continent_counts = defaultdict(int)
    for city in cities:
        continent = get_continent(city['lat'], city['lon'])
        continent_counts[continent] += 1

    print("\nContinent distribution:")
    for continent, count in sorted(continent_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(cities)
        print(f"  {continent}: {count:,} ({pct:.1f}%)")

    # Output path - save to paper/figures
    output_dir = Path(__file__).parent.parent / "paper" / "figures"
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = str(output_dir / "world_map_coverage.png")

    # Plot
    if HAS_CARTOPY:
        plot_world_map_cartopy(cities, output_path)
    else:
        plot_world_map_basic(cities, output_path)


if __name__ == "__main__":
    main()
