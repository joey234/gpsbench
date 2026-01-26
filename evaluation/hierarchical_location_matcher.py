"""
Enhanced Hierarchical Location Matching for Place Association Tasks

This module provides multi-level geographic matching with graduated partial credit:
- Exact location match: 100%
- City match: 75-90%
- Region/State match: 50-60%
- Country match: 30-40%
- Continent match: 15-20%
"""

import re
from typing import Dict, Any, Optional, Tuple
from difflib import SequenceMatcher


# Continent mapping from country codes
COUNTRY_TO_CONTINENT = {
    # Africa
    'DZ': 'Africa', 'AO': 'Africa', 'BJ': 'Africa', 'BW': 'Africa', 'BF': 'Africa', 'BI': 'Africa',
    'CM': 'Africa', 'CV': 'Africa', 'CF': 'Africa', 'TD': 'Africa', 'KM': 'Africa', 'CG': 'Africa',
    'CD': 'Africa', 'CI': 'Africa', 'DJ': 'Africa', 'EG': 'Africa', 'GQ': 'Africa', 'ER': 'Africa',
    'ET': 'Africa', 'GA': 'Africa', 'GM': 'Africa', 'GH': 'Africa', 'GN': 'Africa', 'GW': 'Africa',
    'KE': 'Africa', 'LS': 'Africa', 'LR': 'Africa', 'LY': 'Africa', 'MG': 'Africa', 'MW': 'Africa',
    'ML': 'Africa', 'MR': 'Africa', 'MU': 'Africa', 'YT': 'Africa', 'MA': 'Africa', 'MZ': 'Africa',
    'NA': 'Africa', 'NE': 'Africa', 'NG': 'Africa', 'RE': 'Africa', 'RW': 'Africa', 'SH': 'Africa',
    'ST': 'Africa', 'SN': 'Africa', 'SC': 'Africa', 'SL': 'Africa', 'SO': 'Africa', 'ZA': 'Africa',
    'SS': 'Africa', 'SD': 'Africa', 'SZ': 'Africa', 'TZ': 'Africa', 'TG': 'Africa', 'TN': 'Africa',
    'UG': 'Africa', 'ZM': 'Africa', 'ZW': 'Africa',

    # Asia
    'AF': 'Asia', 'AM': 'Asia', 'AZ': 'Asia', 'BH': 'Asia', 'BD': 'Asia', 'BT': 'Asia',
    'BN': 'Asia', 'KH': 'Asia', 'CN': 'Asia', 'GE': 'Asia', 'HK': 'Asia', 'IN': 'Asia',
    'ID': 'Asia', 'IR': 'Asia', 'IQ': 'Asia', 'IL': 'Asia', 'JP': 'Asia', 'JO': 'Asia',
    'KZ': 'Asia', 'KW': 'Asia', 'KG': 'Asia', 'LA': 'Asia', 'LB': 'Asia', 'MO': 'Asia',
    'MY': 'Asia', 'MV': 'Asia', 'MN': 'Asia', 'MM': 'Asia', 'NP': 'Asia', 'KP': 'Asia',
    'OM': 'Asia', 'PK': 'Asia', 'PS': 'Asia', 'PH': 'Asia', 'QA': 'Asia', 'SA': 'Asia',
    'SG': 'Asia', 'KR': 'Asia', 'LK': 'Asia', 'SY': 'Asia', 'TW': 'Asia', 'TJ': 'Asia',
    'TH': 'Asia', 'TL': 'Asia', 'TR': 'Asia', 'TM': 'Asia', 'AE': 'Asia', 'UZ': 'Asia',
    'VN': 'Asia', 'YE': 'Asia',

    # Europe
    'AX': 'Europe', 'AL': 'Europe', 'AD': 'Europe', 'AT': 'Europe', 'BY': 'Europe', 'BE': 'Europe',
    'BA': 'Europe', 'BG': 'Europe', 'HR': 'Europe', 'CY': 'Europe', 'CZ': 'Europe', 'DK': 'Europe',
    'EE': 'Europe', 'FO': 'Europe', 'FI': 'Europe', 'FR': 'Europe', 'DE': 'Europe', 'GI': 'Europe',
    'GR': 'Europe', 'GG': 'Europe', 'HU': 'Europe', 'IS': 'Europe', 'IE': 'Europe', 'IM': 'Europe',
    'IT': 'Europe', 'JE': 'Europe', 'LV': 'Europe', 'LI': 'Europe', 'LT': 'Europe', 'LU': 'Europe',
    'MK': 'Europe', 'MT': 'Europe', 'MD': 'Europe', 'MC': 'Europe', 'ME': 'Europe', 'NL': 'Europe',
    'NO': 'Europe', 'PL': 'Europe', 'PT': 'Europe', 'RO': 'Europe', 'RU': 'Europe', 'SM': 'Europe',
    'RS': 'Europe', 'SK': 'Europe', 'SI': 'Europe', 'ES': 'Europe', 'SJ': 'Europe', 'SE': 'Europe',
    'CH': 'Europe', 'UA': 'Europe', 'GB': 'Europe', 'VA': 'Europe', 'XK': 'Europe',

    # North America
    'AI': 'North America', 'AG': 'North America', 'AW': 'North America', 'BS': 'North America',
    'BB': 'North America', 'BZ': 'North America', 'BM': 'North America', 'BQ': 'North America',
    'CA': 'North America', 'KY': 'North America', 'CR': 'North America', 'CU': 'North America',
    'CW': 'North America', 'DM': 'North America', 'DO': 'North America', 'SV': 'North America',
    'GL': 'North America', 'GD': 'North America', 'GP': 'North America', 'GT': 'North America',
    'HT': 'North America', 'HN': 'North America', 'JM': 'North America', 'MQ': 'North America',
    'MX': 'North America', 'MS': 'North America', 'NI': 'North America', 'PA': 'North America',
    'PM': 'North America', 'PR': 'North America', 'BL': 'North America', 'KN': 'North America',
    'LC': 'North America', 'MF': 'North America', 'VC': 'North America', 'SX': 'North America',
    'TT': 'North America', 'TC': 'North America', 'US': 'North America', 'VG': 'North America',
    'VI': 'North America',

    # South America
    'AR': 'South America', 'BO': 'South America', 'BR': 'South America', 'CL': 'South America',
    'CO': 'South America', 'EC': 'South America', 'FK': 'South America', 'GF': 'South America',
    'GY': 'South America', 'PY': 'South America', 'PE': 'South America', 'SR': 'South America',
    'UY': 'South America', 'VE': 'South America',

    # Oceania
    'AS': 'Oceania', 'AU': 'Oceania', 'CX': 'Oceania', 'CC': 'Oceania', 'CK': 'Oceania',
    'FJ': 'Oceania', 'PF': 'Oceania', 'GU': 'Oceania', 'KI': 'Oceania', 'MH': 'Oceania',
    'FM': 'Oceania', 'NR': 'Oceania', 'NC': 'Oceania', 'NZ': 'Oceania', 'NU': 'Oceania',
    'NF': 'Oceania', 'MP': 'Oceania', 'PW': 'Oceania', 'PG': 'Oceania', 'PN': 'Oceania',
    'WS': 'Oceania', 'SB': 'Oceania', 'TK': 'Oceania', 'TO': 'Oceania', 'TV': 'Oceania',
    'VU': 'Oceania', 'WF': 'Oceania',

    # Antarctica
    'AQ': 'Antarctica', 'BV': 'Antarctica', 'GS': 'Antarctica', 'HM': 'Antarctica', 'TF': 'Antarctica'
}


# Country code to full name
COUNTRY_CODE_TO_NAME = {
    'US': 'United States', 'GB': 'United Kingdom', 'CN': 'China', 'IN': 'India',
    'JP': 'Japan', 'DE': 'Germany', 'FR': 'France', 'BR': 'Brazil', 'IT': 'Italy',
    'CA': 'Canada', 'RU': 'Russia', 'KR': 'South Korea', 'ES': 'Spain', 'MX': 'Mexico',
    'AU': 'Australia', 'ID': 'Indonesia', 'NL': 'Netherlands', 'SA': 'Saudi Arabia',
    'TR': 'Turkey', 'CH': 'Switzerland', 'PL': 'Poland', 'BE': 'Belgium', 'SE': 'Sweden',
    'NG': 'Nigeria', 'AR': 'Argentina', 'NO': 'Norway', 'AT': 'Austria', 'AE': 'United Arab Emirates',
    'IL': 'Israel', 'HK': 'Hong Kong', 'IE': 'Ireland', 'SG': 'Singapore', 'DK': 'Denmark',
    'MY': 'Malaysia', 'CO': 'Colombia', 'PH': 'Philippines', 'PK': 'Pakistan', 'CL': 'Chile',
    'FI': 'Finland', 'BD': 'Bangladesh', 'EG': 'Egypt', 'VN': 'Vietnam', 'CZ': 'Czech Republic',
    'PT': 'Portugal', 'RO': 'Romania', 'NZ': 'New Zealand', 'IR': 'Iran', 'TH': 'Thailand',
    'ZA': 'South Africa', 'GR': 'Greece', 'HU': 'Hungary', 'UA': 'Ukraine', 'DZ': 'Algeria',
    # Add more as needed
}


class HierarchicalLocationMatcher:
    """
    Matches location responses with graduated partial credit
    """

    def __init__(self):
        pass

    def extract_location_hierarchy(self, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract hierarchical location information from ground truth

        Returns dict with keys: location, city, region, country, continent
        """
        hierarchy = {
            'location': None,
            'city': None,
            'region': None,
            'country': None,
            'continent': None
        }

        # Get full location name - support both 'name' (old) and 'location_name' (new)
        location_name = ground_truth.get('location_name') or ground_truth.get('name', '')
        hierarchy['location'] = location_name

        # Parse compound names like "Times Square, Manhattan, New York"
        parts = [p.strip() for p in location_name.split(',')]

        # If landmark/specific location, extract city from parts
        if len(parts) >= 2 and ground_truth.get('type') == 'landmark':
            # e.g., "Eiffel Tower, Paris" → city is "Paris"
            hierarchy['city'] = parts[1] if len(parts) >= 2 else None

        elif ground_truth.get('type') == 'city':
            # For city tests, the name itself is the city
            hierarchy['city'] = parts[0]

        # Get region if available
        if 'region' in ground_truth:
            hierarchy['region'] = ground_truth['region']

        # Get country
        country_code = ground_truth.get('country') or ground_truth.get('country_code')
        if country_code:
            hierarchy['country'] = country_code
            hierarchy['country_name'] = COUNTRY_CODE_TO_NAME.get(country_code, country_code)

            # Get continent from country code
            hierarchy['continent'] = COUNTRY_TO_CONTINENT.get(country_code)

        return hierarchy

    def normalize_location_name(self, name: str) -> str:
        """Normalize location name for matching"""
        if not name:
            return ""

        # Remove common artifacts
        name = re.sub(r'\(.*?\)', '', name)  # Remove parentheticals
        name = name.strip().lower()

        # Remove common suffixes
        name = re.sub(r',?\s+(city|province|state|region)$', '', name)

        return name

    def match_with_hierarchy(self, response: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        Match response against hierarchical location levels

        Returns:
            Dictionary with separate scores for each level:
            {
                'exact_location_match': bool,
                'city_match': bool,
                'region_match': bool,
                'country_match': bool,
                'continent_match': bool,
                'location_name': str (what was matched),
                'match_notes': str
            }
        """
        hierarchy = self.extract_location_hierarchy(ground_truth)
        response_lower = response.lower()

        # Initialize all levels as False
        result = {
            'exact_location_match': False,
            'city_match': False,
            'region_match': False,
            'country_match': False,
            'continent_match': False,
            'location_name': None,
            'city_name': None,
            'region_name': None,
            'country_name': None,
            'continent_name': None,
            'match_notes': []
        }

        # Level 1: Exact location match
        location = self.normalize_location_name(hierarchy['location'])
        if location and location in response_lower:
            result['exact_location_match'] = True
            result['location_name'] = hierarchy['location']
            result['match_notes'].append(f"Exact location: {hierarchy['location']}")

        # Level 2: City match
        if hierarchy['city']:
            city = self.normalize_location_name(hierarchy['city'])
            if city and city in response_lower:
                result['city_match'] = True
                result['city_name'] = hierarchy['city']
                result['match_notes'].append(f"City: {hierarchy['city']}")

        # Level 3: Region match
        if hierarchy['region']:
            region = self.normalize_location_name(hierarchy['region'])
            if region and len(region) > 3 and region in response_lower:
                result['region_match'] = True
                result['region_name'] = hierarchy['region']
                result['match_notes'].append(f"Region: {hierarchy['region']}")

        # Level 4: Country match
        if hierarchy['country']:
            country_code = hierarchy['country'].lower()
            country_name = hierarchy.get('country_name', '').lower()

            # Check both country code and full name
            if country_code in response_lower or country_name in response_lower:
                result['country_match'] = True
                result['country_name'] = hierarchy.get('country_name', hierarchy['country'])
                result['match_notes'].append(f"Country: {result['country_name']}")

        # Level 5: Continent match
        if hierarchy['continent']:
            continent = hierarchy['continent'].lower()
            if continent and continent in response_lower:
                result['continent_match'] = True
                result['continent_name'] = hierarchy['continent']
                result['match_notes'].append(f"Continent: {hierarchy['continent']}")

        # Add overall assessment
        result['any_match'] = any([
            result['exact_location_match'],
            result['city_match'],
            result['region_match'],
            result['country_match'],
            result['continent_match']
        ])

        result['match_notes'] = '; '.join(result['match_notes']) if result['match_notes'] else 'No match'

        return result


def demo():
    """Demonstrate hierarchical matching with separate scores"""
    matcher = HierarchicalLocationMatcher()

    # Test case 1: Landmark
    ground_truth_landmark = {
        'type': 'landmark',
        'name': 'Eiffel Tower, Paris',
        'country': 'FR',
        'region': 'Île-de-France'
    }

    test_responses = [
        'Eiffel Tower',
        'Paris',
        'France',
        'Europe',
        'Île-de-France',
        'London'
    ]

    print("Test Case: Eiffel Tower")
    print("="*70)
    for resp in test_responses:
        result = matcher.match_with_hierarchy(resp, ground_truth_landmark)
        print(f"\nResponse: '{resp}'")
        print(f"  Exact: {result['exact_location_match']}")
        print(f"  City: {result['city_match']}")
        print(f"  Region: {result['region_match']}")
        print(f"  Country: {result['country_match']}")
        print(f"  Continent: {result['continent_match']}")
        print(f"  Notes: {result['match_notes']}")

    print("\n" + "="*70)

    # Test case 2: City
    ground_truth_city = {
        'type': 'city',
        'name': 'New York',
        'country_code': 'US'
    }

    test_responses = [
        'New York City',
        'United States',
        'North America',
        'Tokyo'
    ]

    print("\nTest Case: New York")
    print("="*70)
    for resp in test_responses:
        result = matcher.match_with_hierarchy(resp, ground_truth_city)
        print(f"\nResponse: '{resp}'")
        print(f"  Exact: {result['exact_location_match']}")
        print(f"  City: {result['city_match']}")
        print(f"  Region: {result['region_match']}")
        print(f"  Country: {result['country_match']}")
        print(f"  Continent: {result['continent_match']}")
        print(f"  Notes: {result['match_notes']}")


if __name__ == '__main__':
    demo()
