#!/usr/bin/env python3
"""
Augment place_association data with geographic hierarchy metadata.

Adds province/state, country name, and region/continent information to each sample
so downstream analysis can break down results by geographic level.

Usage:
    python scripts/augment_place_association_hierarchy.py
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional

# Country code to full name mapping
COUNTRY_NAMES = {
    'US': 'United States', 'GB': 'United Kingdom', 'CN': 'China', 'IN': 'India',
    'JP': 'Japan', 'DE': 'Germany', 'FR': 'France', 'BR': 'Brazil', 'IT': 'Italy',
    'CA': 'Canada', 'RU': 'Russia', 'KR': 'South Korea', 'ES': 'Spain', 'MX': 'Mexico',
    'AU': 'Australia', 'ID': 'Indonesia', 'NL': 'Netherlands', 'SA': 'Saudi Arabia',
    'TR': 'Turkey', 'CH': 'Switzerland', 'PL': 'Poland', 'BE': 'Belgium', 'SE': 'Sweden',
    'NG': 'Nigeria', 'AR': 'Argentina', 'NO': 'Norway', 'AT': 'Austria', 'AE': 'UAE',
    'IL': 'Israel', 'HK': 'Hong Kong', 'IE': 'Ireland', 'SG': 'Singapore', 'DK': 'Denmark',
    'MY': 'Malaysia', 'CO': 'Colombia', 'PH': 'Philippines', 'PK': 'Pakistan', 'CL': 'Chile',
    'FI': 'Finland', 'BD': 'Bangladesh', 'EG': 'Egypt', 'VN': 'Vietnam', 'CZ': 'Czech Republic',
    'PT': 'Portugal', 'RO': 'Romania', 'NZ': 'New Zealand', 'IR': 'Iran', 'TH': 'Thailand',
    'ZA': 'South Africa', 'GR': 'Greece', 'HU': 'Hungary', 'UA': 'Ukraine', 'DZ': 'Algeria',
    'GT': 'Guatemala', 'PE': 'Peru', 'EC': 'Ecuador', 'VE': 'Venezuela', 'CU': 'Cuba',
    'BO': 'Bolivia', 'DO': 'Dominican Republic', 'HN': 'Honduras', 'PY': 'Paraguay',
    'SV': 'El Salvador', 'NI': 'Nicaragua', 'CR': 'Costa Rica', 'PA': 'Panama', 'UY': 'Uruguay',
    'KE': 'Kenya', 'ET': 'Ethiopia', 'TZ': 'Tanzania', 'UG': 'Uganda', 'GH': 'Ghana',
    'CI': 'Ivory Coast', 'CM': 'Cameroon', 'MG': 'Madagascar', 'AO': 'Angola', 'MZ': 'Mozambique',
    'NE': 'Niger', 'ML': 'Mali', 'BF': 'Burkina Faso', 'MW': 'Malawi', 'ZM': 'Zambia',
    'ZW': 'Zimbabwe', 'SN': 'Senegal', 'TD': 'Chad', 'SO': 'Somalia', 'GN': 'Guinea',
    'RW': 'Rwanda', 'BJ': 'Benin', 'TN': 'Tunisia', 'SS': 'South Sudan', 'TG': 'Togo',
    'SL': 'Sierra Leone', 'LY': 'Libya', 'CG': 'Congo', 'LR': 'Liberia', 'CF': 'Central African Republic',
    'MR': 'Mauritania', 'ER': 'Eritrea', 'GM': 'Gambia', 'BW': 'Botswana', 'NA': 'Namibia',
    'GA': 'Gabon', 'LS': 'Lesotho', 'GW': 'Guinea-Bissau', 'MU': 'Mauritius', 'SZ': 'Eswatini',
    'DJ': 'Djibouti', 'RE': 'Reunion', 'KM': 'Comoros', 'CV': 'Cape Verde', 'ST': 'Sao Tome and Principe',
    'SC': 'Seychelles', 'MM': 'Myanmar', 'AF': 'Afghanistan', 'NP': 'Nepal', 'KH': 'Cambodia',
    'LA': 'Laos', 'LK': 'Sri Lanka', 'KZ': 'Kazakhstan', 'UZ': 'Uzbekistan', 'TM': 'Turkmenistan',
    'KG': 'Kyrgyzstan', 'TJ': 'Tajikistan', 'AZ': 'Azerbaijan', 'GE': 'Georgia', 'AM': 'Armenia',
    'MN': 'Mongolia', 'BN': 'Brunei', 'TL': 'Timor-Leste', 'BT': 'Bhutan', 'MV': 'Maldives',
    'IQ': 'Iraq', 'SY': 'Syria', 'JO': 'Jordan', 'LB': 'Lebanon', 'PS': 'Palestine',
    'YE': 'Yemen', 'OM': 'Oman', 'KW': 'Kuwait', 'BH': 'Bahrain', 'QA': 'Qatar',
    'CY': 'Cyprus', 'AL': 'Albania', 'MK': 'North Macedonia', 'RS': 'Serbia', 'BA': 'Bosnia and Herzegovina',
    'HR': 'Croatia', 'SI': 'Slovenia', 'SK': 'Slovakia', 'LT': 'Lithuania', 'LV': 'Latvia',
    'EE': 'Estonia', 'BY': 'Belarus', 'MD': 'Moldova', 'BG': 'Bulgaria', 'ME': 'Montenegro',
    'XK': 'Kosovo', 'IS': 'Iceland', 'LU': 'Luxembourg', 'MT': 'Malta', 'AD': 'Andorra',
    'MC': 'Monaco', 'SM': 'San Marino', 'LI': 'Liechtenstein', 'VA': 'Vatican City',
    'FJ': 'Fiji', 'PG': 'Papua New Guinea', 'SB': 'Solomon Islands', 'VU': 'Vanuatu',
    'NC': 'New Caledonia', 'PF': 'French Polynesia', 'WS': 'Samoa', 'GU': 'Guam',
    'KI': 'Kiribati', 'FM': 'Micronesia', 'TO': 'Tonga', 'PW': 'Palau', 'MH': 'Marshall Islands',
    'TV': 'Tuvalu', 'NR': 'Nauru', 'JM': 'Jamaica', 'TT': 'Trinidad and Tobago', 'BS': 'Bahamas',
    'BB': 'Barbados', 'LC': 'Saint Lucia', 'GD': 'Grenada', 'VC': 'Saint Vincent', 'AG': 'Antigua and Barbuda',
    'DM': 'Dominica', 'KN': 'Saint Kitts and Nevis', 'HT': 'Haiti', 'BZ': 'Belize', 'SR': 'Suriname',
    'GY': 'Guyana', 'TW': 'Taiwan', 'PR': 'Puerto Rico', 'KP': 'North Korea', 'CD': 'DR Congo',
    'SD': 'Sudan', 'MA': 'Morocco',
}

# Country to region/continent mapping
COUNTRY_REGIONS = {
    # North America
    'US': 'North America', 'CA': 'North America', 'MX': 'North America',
    # Central America
    'GT': 'Central America', 'BZ': 'Central America', 'SV': 'Central America',
    'HN': 'Central America', 'NI': 'Central America', 'CR': 'Central America', 'PA': 'Central America',
    # Caribbean
    'CU': 'Caribbean', 'JM': 'Caribbean', 'HT': 'Caribbean', 'DO': 'Caribbean', 'PR': 'Caribbean',
    'BS': 'Caribbean', 'BB': 'Caribbean', 'TT': 'Caribbean', 'LC': 'Caribbean', 'GD': 'Caribbean',
    'VC': 'Caribbean', 'AG': 'Caribbean', 'DM': 'Caribbean', 'KN': 'Caribbean',
    # South America
    'BR': 'South America', 'AR': 'South America', 'CO': 'South America', 'PE': 'South America',
    'VE': 'South America', 'CL': 'South America', 'EC': 'South America', 'BO': 'South America',
    'PY': 'South America', 'UY': 'South America', 'GY': 'South America', 'SR': 'South America',
    # Western Europe
    'GB': 'Western Europe', 'DE': 'Western Europe', 'FR': 'Western Europe', 'NL': 'Western Europe',
    'BE': 'Western Europe', 'CH': 'Western Europe', 'AT': 'Western Europe', 'IE': 'Western Europe',
    'LU': 'Western Europe', 'MC': 'Western Europe', 'LI': 'Western Europe',
    # Southern Europe
    'IT': 'Southern Europe', 'ES': 'Southern Europe', 'PT': 'Southern Europe', 'GR': 'Southern Europe',
    'MT': 'Southern Europe', 'AD': 'Southern Europe', 'SM': 'Southern Europe', 'VA': 'Southern Europe',
    'CY': 'Southern Europe', 'AL': 'Southern Europe', 'MK': 'Southern Europe', 'RS': 'Southern Europe',
    'BA': 'Southern Europe', 'HR': 'Southern Europe', 'SI': 'Southern Europe', 'ME': 'Southern Europe', 'XK': 'Southern Europe',
    # Northern Europe
    'SE': 'Northern Europe', 'NO': 'Northern Europe', 'DK': 'Northern Europe', 'FI': 'Northern Europe',
    'IS': 'Northern Europe', 'EE': 'Northern Europe', 'LV': 'Northern Europe', 'LT': 'Northern Europe',
    # Eastern Europe
    'PL': 'Eastern Europe', 'CZ': 'Eastern Europe', 'SK': 'Eastern Europe', 'HU': 'Eastern Europe',
    'RO': 'Eastern Europe', 'BG': 'Eastern Europe', 'UA': 'Eastern Europe', 'BY': 'Eastern Europe',
    'MD': 'Eastern Europe', 'RU': 'Eastern Europe',
    # East Asia
    'CN': 'East Asia', 'JP': 'East Asia', 'KR': 'East Asia', 'KP': 'East Asia',
    'TW': 'East Asia', 'MN': 'East Asia', 'HK': 'East Asia',
    # Southeast Asia
    'ID': 'Southeast Asia', 'TH': 'Southeast Asia', 'VN': 'Southeast Asia', 'MY': 'Southeast Asia',
    'PH': 'Southeast Asia', 'SG': 'Southeast Asia', 'MM': 'Southeast Asia', 'KH': 'Southeast Asia',
    'LA': 'Southeast Asia', 'BN': 'Southeast Asia', 'TL': 'Southeast Asia',
    # South Asia
    'IN': 'South Asia', 'PK': 'South Asia', 'BD': 'South Asia', 'NP': 'South Asia',
    'LK': 'South Asia', 'BT': 'South Asia', 'MV': 'South Asia', 'AF': 'South Asia',
    # Central Asia
    'KZ': 'Central Asia', 'UZ': 'Central Asia', 'TM': 'Central Asia',
    'KG': 'Central Asia', 'TJ': 'Central Asia',
    # Caucasus
    'AZ': 'Caucasus', 'GE': 'Caucasus', 'AM': 'Caucasus',
    # Middle East
    'SA': 'Middle East', 'AE': 'Middle East', 'IR': 'Middle East', 'IQ': 'Middle East',
    'IL': 'Middle East', 'JO': 'Middle East', 'LB': 'Middle East', 'SY': 'Middle East',
    'YE': 'Middle East', 'OM': 'Middle East', 'KW': 'Middle East', 'BH': 'Middle East',
    'QA': 'Middle East', 'PS': 'Middle East', 'TR': 'Middle East',
    # North Africa
    'EG': 'North Africa', 'LY': 'North Africa', 'TN': 'North Africa', 'DZ': 'North Africa',
    'MA': 'North Africa', 'MR': 'North Africa', 'SD': 'North Africa',
    # West Africa
    'NG': 'West Africa', 'GH': 'West Africa', 'CI': 'West Africa', 'SN': 'West Africa',
    'ML': 'West Africa', 'BF': 'West Africa', 'NE': 'West Africa', 'GN': 'West Africa',
    'BJ': 'West Africa', 'TG': 'West Africa', 'SL': 'West Africa', 'LR': 'West Africa',
    'GM': 'West Africa', 'GW': 'West Africa', 'CV': 'West Africa',
    # East Africa
    'ET': 'East Africa', 'KE': 'East Africa', 'TZ': 'East Africa', 'UG': 'East Africa',
    'RW': 'East Africa', 'SO': 'East Africa', 'ER': 'East Africa', 'DJ': 'East Africa',
    'SS': 'East Africa', 'MW': 'East Africa', 'MZ': 'East Africa', 'MG': 'East Africa',
    'MU': 'East Africa', 'KM': 'East Africa', 'SC': 'East Africa', 'RE': 'East Africa',
    # Southern Africa
    'ZA': 'Southern Africa', 'ZW': 'Southern Africa', 'ZM': 'Southern Africa', 'BW': 'Southern Africa',
    'NA': 'Southern Africa', 'AO': 'Southern Africa', 'LS': 'Southern Africa', 'SZ': 'Southern Africa',
    # Central Africa
    'CM': 'Central Africa', 'CD': 'Central Africa', 'CG': 'Central Africa', 'GA': 'Central Africa',
    'CF': 'Central Africa', 'TD': 'Central Africa', 'ST': 'Central Africa',
    # Oceania
    'AU': 'Oceania', 'NZ': 'Oceania', 'PG': 'Oceania', 'FJ': 'Oceania', 'SB': 'Oceania',
    'VU': 'Oceania', 'NC': 'Oceania', 'PF': 'Oceania', 'WS': 'Oceania', 'GU': 'Oceania',
    'KI': 'Oceania', 'FM': 'Oceania', 'TO': 'Oceania', 'PW': 'Oceania', 'MH': 'Oceania',
    'TV': 'Oceania', 'NR': 'Oceania',
}


class GeoNamesLookup:
    """Lookup geographic hierarchy from GeoNames data."""

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.admin1_codes = {}
        self.cities_index = {}  # (name, country_code) -> city info
        self.coords_index = defaultdict(list)  # rounded coords -> cities

    def load(self):
        """Load GeoNames data files."""
        self._load_admin1_codes()
        self._load_cities()
        print(f"Loaded {len(self.admin1_codes)} admin1 codes")
        print(f"Loaded {len(self.cities_index)} unique city entries")

    def _load_admin1_codes(self):
        """Load admin1 (state/province) codes."""
        admin1_file = self.data_path / "raw" / "admin1CodesASCII.txt"
        with open(admin1_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    self.admin1_codes[parts[0]] = parts[1]

    def _load_cities(self):
        """Load cities and build lookup indexes."""
        cities_file = self.data_path / "raw" / "cities15000.txt"
        with open(cities_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 15:
                    continue

                name = parts[1]
                ascii_name = parts[2]
                alt_names = parts[3].split(',') if parts[3] else []
                lat = float(parts[4])
                lon = float(parts[5])
                country_code = parts[8]
                admin1_code = parts[10]

                city_info = {
                    'name': name,
                    'ascii_name': ascii_name,
                    'lat': lat,
                    'lon': lon,
                    'country_code': country_code,
                    'admin1_code': admin1_code,
                }

                # Index by name and country
                key = (name.lower(), country_code)
                self.cities_index[key] = city_info

                # Also index by ascii name
                key_ascii = (ascii_name.lower(), country_code)
                if key_ascii not in self.cities_index:
                    self.cities_index[key_ascii] = city_info

                # Index alternate names
                for alt in alt_names:
                    alt_key = (alt.lower().strip(), country_code)
                    if alt_key not in self.cities_index:
                        self.cities_index[alt_key] = city_info

                # Index by rounded coordinates for fallback lookup
                coord_key = (round(lat, 2), round(lon, 2))
                self.coords_index[coord_key].append(city_info)

    def lookup(self, location_name: str, country_code: str, lat: float = None, lon: float = None) -> Dict:
        """Look up geographic hierarchy for a location."""
        result = {
            'admin1_name': None,
            'country_name': COUNTRY_NAMES.get(country_code, country_code),
            'region': COUNTRY_REGIONS.get(country_code, 'Unknown'),
        }

        # Try direct name lookup
        key = (location_name.lower(), country_code)
        city_info = self.cities_index.get(key)

        # Try coordinate-based fallback if name not found
        if not city_info and lat is not None and lon is not None:
            coord_key = (round(lat, 2), round(lon, 2))
            candidates = self.coords_index.get(coord_key, [])
            # Find closest match
            if candidates:
                min_dist = float('inf')
                for c in candidates:
                    if c['country_code'] == country_code:
                        dist = abs(c['lat'] - lat) + abs(c['lon'] - lon)
                        if dist < min_dist:
                            min_dist = dist
                            city_info = c

        # Get admin1 name if found
        if city_info and city_info['admin1_code']:
            admin1_key = f"{country_code}.{city_info['admin1_code']}"
            result['admin1_name'] = self.admin1_codes.get(admin1_key)

        return result


def augment_place_association_data(data_path: Path):
    """Augment place_association data with geographic hierarchy."""

    # Initialize GeoNames lookup
    print("Loading GeoNames data...")
    lookup = GeoNamesLookup(data_path)
    lookup.load()
    print()

    # Process main file and splits
    files_to_process = [
        data_path / "track_applied" / "task1_place_association.json",
        data_path / "track_applied" / "splits" / "task1_place_association_train.json",
        data_path / "track_applied" / "splits" / "task1_place_association_dev.json",
        data_path / "track_applied" / "splits" / "task1_place_association_test.json",
    ]

    for file_path in files_to_process:
        if not file_path.exists():
            print(f"Skipping {file_path} (not found)")
            continue

        print(f"Processing {file_path.name}...")

        with open(file_path, 'r') as f:
            data = json.load(f)

        augmented_count = 0
        missing_admin1 = 0

        for sample in data:
            gt = sample.get('ground_truth', {})
            coord = sample.get('coordinate', {})

            location_name = gt.get('location_name', '')
            country_code = gt.get('country_code', '')
            lat = coord.get('true_lat', coord.get('lat'))
            lon = coord.get('true_lon', coord.get('lon'))

            # Look up hierarchy
            hierarchy = lookup.lookup(location_name, country_code, lat, lon)

            # Add to ground_truth
            gt['admin1_name'] = hierarchy['admin1_name']
            gt['country_name'] = hierarchy['country_name']
            gt['region'] = hierarchy['region']

            if hierarchy['admin1_name']:
                augmented_count += 1
            else:
                missing_admin1 += 1

        # Save augmented data
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"  Total: {len(data)} samples")
        print(f"  With admin1: {augmented_count}")
        print(f"  Missing admin1: {missing_admin1}")
        print()

    print("Done!")


def print_statistics(data_path: Path):
    """Print statistics about the augmented data."""
    from collections import Counter

    file_path = data_path / "track_applied" / "task1_place_association.json"
    with open(file_path, 'r') as f:
        data = json.load(f)

    print("\n" + "=" * 70)
    print("Geographic Distribution Statistics")
    print("=" * 70)

    # By region
    regions = Counter(s['ground_truth'].get('region', 'Unknown') for s in data)
    print("\nBy Region:")
    for region, count in sorted(regions.items(), key=lambda x: -x[1]):
        pct = count / len(data) * 100
        print(f"  {region}: {count} ({pct:.1f}%)")

    # By country (top 15)
    countries = Counter(s['ground_truth'].get('country_name', 'Unknown') for s in data)
    print("\nTop 15 Countries:")
    for country, count in countries.most_common(15):
        pct = count / len(data) * 100
        print(f"  {country}: {count} ({pct:.1f}%)")

    # Admin1 coverage
    with_admin1 = sum(1 for s in data if s['ground_truth'].get('admin1_name'))
    print(f"\nAdmin1 Coverage: {with_admin1}/{len(data)} ({with_admin1/len(data)*100:.1f}%)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Augment place_association data with geographic hierarchy")
    parser.add_argument('--data-path', type=str, default='data', help='Path to data directory')
    parser.add_argument('--stats-only', action='store_true', help='Only print statistics, do not modify data')
    args = parser.parse_args()

    data_path = Path(args.data_path)

    if args.stats_only:
        print_statistics(data_path)
    else:
        augment_place_association_data(data_path)
        print_statistics(data_path)


if __name__ == "__main__":
    main()
