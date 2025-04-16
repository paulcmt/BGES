import os
import pandas as pd
from datetime import datetime
import psycopg2
from sqlalchemy import create_engine
from typing import Dict, List, Any, Optional, Tuple
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import ssl
import certifi
import time
from functools import lru_cache

from constants import *

# Ignore timezone for pyarrow
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

# Initialize geocoder with SSL context
ctx = ssl.create_default_context(cafile=certifi.where())
ctx.verify_mode = ssl.CERT_REQUIRED
geolocator = Nominatim(user_agent="bges_etl", ssl_context=ctx)

# Cache for city coordinates
@lru_cache(maxsize=1000)
def get_city_coordinates(city: str, country: str) -> Optional[Tuple[float, float]]:
    """
    Get latitude and longitude coordinates for a city with caching
    """
    try:
        # Standardize city and country names
        city = CITY_MAPPING.get(city.upper(), city)
        country = COUNTRY_MAPPING.get(country, country)
        
        location = geolocator.geocode(f"{city}, {country}", timeout=10)
        if location:
            return (location.latitude, location.longitude)
        else:
            print(f"Could not find coordinates for {city}, {country}")
            return None
    except GeocoderTimedOut:
        print(f"Timeout getting coordinates for {city}, {country}")
        return None
    except Exception as e:
        print(f"Error getting coordinates for {city}, {country}: {str(e)}")
        return None

def calculate_distance(departure_city: str, departure_country: str,
                      destination_city: str, destination_country: str) -> float:
    """
    Calculate distance in kilometers between two cities
    """
    # Get coordinates for both cities with retry
    departure_coords = get_city_coordinates(departure_city, departure_country)
    if not departure_coords:
        time.sleep(1)  # Wait a bit and retry once
        departure_coords = get_city_coordinates(departure_city, departure_country)
        
    destination_coords = get_city_coordinates(destination_city, destination_country)
    if not destination_coords:
        time.sleep(1)  # Wait a bit and retry once
        destination_coords = get_city_coordinates(destination_city, destination_country)
    
    if departure_coords and destination_coords:
        # Calculate distance in kilometers
        distance = geodesic(departure_coords, destination_coords).kilometers
        return round(distance, 2)
    else:
        print(f"Could not calculate distance between {departure_city} and {destination_city}")
        return 0.0

def extract_city_data(city):
    """
    Extract data for a specific city from all relevant files
    """
    city_data = {}
    
    # Extract personnel data
    personnel_path = f"{BASE_PATH}/BDD_BGES_{city}/PERSONNEL_{city}.txt"
    if os.path.exists(personnel_path):
        personnel_df = pd.read_csv(personnel_path, delimiter=';')
        city_data['personnel'] = personnel_df
        print(f"{SUCCESS}✓ Successfully extracted personnel data for {city}{RESET}")
    
    # Extract informatique data
    informatique_dir = f"{BASE_PATH}/BDD_BGES_{city}/BDD_BGES_{city}_INFORMATIQUE"
    if os.path.exists(informatique_dir):
        informatique_files = [f for f in os.listdir(informatique_dir) 
                            if f.startswith('MATERIEL_INFORMATIQUE_') and f.endswith('.txt')]
        informatique_dfs = []
        for file in informatique_files:
            df = pd.read_csv(os.path.join(informatique_dir, file), delimiter=';')
            informatique_dfs.append(df)
        if informatique_dfs:
            city_data['informatique'] = pd.concat(informatique_dfs, ignore_index=True)
            print(f"{SUCCESS}✓ Successfully extracted informatique data for {city}{RESET}")
    
    # Extract mission data
    mission_dir = f"{BASE_PATH}/BDD_BGES_{city}/BDD_BGES_{city}_MISSION"
    if os.path.exists(mission_dir):
        mission_files = [f for f in os.listdir(mission_dir) 
                        if f.startswith('MISSION_') and f.endswith('.txt')]
        mission_dfs = []
        for file in mission_files:
            df = pd.read_csv(os.path.join(mission_dir, file), delimiter=';')
            mission_dfs.append(df)
        if mission_dfs:
            city_data['mission'] = pd.concat(mission_dfs, ignore_index=True)
            print(f"{SUCCESS}✓ Successfully extracted mission data for {city}{RESET}")
    
    return city_data

def extract_data():
    """
    Extract data for all cities
    """
    city_data_dict = {}
    print(f"{INFO}Starting data extraction for all cities...{RESET}")
    for city in CITIES:
        print(f"{INFO}Extracting data for {city}...{RESET}")
        city_data = extract_city_data(city)
        if city_data:
            city_data_dict[city] = city_data
            print(f"{SUCCESS}✓ Successfully completed extraction for {city}{RESET}")
        else:
            print(f"{WARNING}No data found for {city}{RESET}")
    
    print(f"{SUCCESS}✓ Data extraction completed successfully{RESET}")
    return city_data_dict

def transform_mission_data(mission_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform mission data to match fact_business_travel schema
    """
    print(f"{INFO}Starting mission data transformation...{RESET}")
    
    # Rename columns to match database schema
    transformed = mission_df.rename(columns={
        'ID_MISSION': 'TRAVEL_ID',
        'ID_PERSONNEL': 'EMPLOYEE_ID',
        'DATE_MISSION': 'DATE_ID',
        'TYPE_MISSION': 'MISSION_TYPE_ID',
        'VILLE_DEPART': 'DEPARTURE_CITY',
        'PAYS_DEPART': 'DEPARTURE_COUNTRY',
        'VILLE_DESTINATION': 'DESTINATION_CITY',
        'PAYS_DESTINATION': 'DESTINATION_COUNTRY',
        'TRANSPORT': 'TRANSPORT_ID',
        'ALLER_RETOUR': 'IS_ROUND_TRIP'
    })
    
    # Convert date to datetime
    transformed['DATE_ID'] = pd.to_datetime(transformed['DATE_ID'])
    print(f"{SUCCESS}✓ Date conversion completed{RESET}")
    
    # Convert boolean values
    transformed['IS_ROUND_TRIP'] = transformed['IS_ROUND_TRIP'].map({'oui': True, 'non': False})
    print(f"{SUCCESS}✓ Boolean conversion completed{RESET}")
    
    # Translate mission types to English
    transformed['MISSION_TYPE_ID'] = transformed['MISSION_TYPE_ID'].str.lower().map(MISSION_TYPE_TRANSLATIONS)
    print(f"{SUCCESS}✓ Mission type translation completed{RESET}")
    
    # Calculate distances with progress indicator
    print(f"{INFO}Calculating distances between cities...{RESET}")
    total_rows = len(transformed)
    transformed['DISTANCE_KM'] = transformed.apply(
        lambda row: calculate_distance(
            row['DEPARTURE_CITY'],
            row['DEPARTURE_COUNTRY'],
            row['DESTINATION_CITY'],
            row['DESTINATION_COUNTRY']
        ),
        axis=1
    )
    print(f"{SUCCESS}✓ Distance calculation completed{RESET}")
    
    # Double the distance for round trips
    transformed.loc[transformed['IS_ROUND_TRIP'], 'DISTANCE_KM'] *= 2
    print(f"{SUCCESS}✓ Round trip distance adjustment completed{RESET}")
    
    # Placeholder for emissions (will be implemented later)
    transformed['EMISSIONS_KG_CO2E'] = 0
    
    # Select only the columns we need
    required_columns = [
        'TRAVEL_ID', 'EMPLOYEE_ID', 'MISSION_TYPE_ID', 'DEPARTURE_CITY',
        'DEPARTURE_COUNTRY', 'DESTINATION_CITY', 'DESTINATION_COUNTRY',
        'TRANSPORT_ID', 'DATE_ID', 'DISTANCE_KM', 'IS_ROUND_TRIP',
        'EMISSIONS_KG_CO2E'
    ]
    transformed = transformed[required_columns]
    print(f"{SUCCESS}✓ Column selection completed{RESET}")
    
    print(f"{SUCCESS}✓ Mission data transformation completed successfully{RESET}")
    return transformed

def transform_all_mission_data(extracted_data: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Transform mission data from all cities into a single DataFrame
    """
    mission_dfs = []
    print(f"{INFO}Starting transformation of all mission data...{RESET}")
    
    for city, city_data in extracted_data.items():
        if 'mission' in city_data:
            print(f"{INFO}Transforming mission data for {city}...{RESET}")
            transformed_df = transform_mission_data(city_data['mission'])
            mission_dfs.append(transformed_df)
            print(f"{SUCCESS}✓ Successfully transformed mission data for {city}{RESET}")
    
    if mission_dfs:
        result = pd.concat(mission_dfs, ignore_index=True)
        print(f"{SUCCESS}✓ All mission data transformation completed successfully{RESET}")
        return result
    else:
        print(f"{WARNING}No mission data found to transform{RESET}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Test the extraction and transformation
    print(f"{INFO}Starting ETL process...{RESET}")
    print("\nExtracting data...")
    data = extract_data()
    
    print("\nTransforming mission data...")
    transformed_missions = transform_all_mission_data(data)
    
    print(f"\n{SUCCESS}ETL process completed successfully{RESET}")
    print("\nTransformed mission data summary:")
    print(f"Total number of missions: {len(transformed_missions)}")
    print("\nFirst few rows of transformed data:")
    print(transformed_missions.head())
