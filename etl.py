import os
import pandas as pd
from datetime import datetime
import psycopg2
from sqlalchemy import create_engine as sqlalchemy_create_engine
from typing import Dict, List, Any, Optional, Tuple
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import ssl
import certifi
import time
from functools import lru_cache
from sqlalchemy import text
from dotenv import load_dotenv

from constants import *

# Load environment variables from db.env
load_dotenv('db.env')

# Database connection parameters from environment variables
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')

# Ignore timezone for pyarrow
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

# Initialize geocoder with SSL context
ctx = ssl.create_default_context(cafile=certifi.where())
ctx.verify_mode = ssl.CERT_REQUIRED
geolocator = Nominatim(user_agent="bges_etl", ssl_context=ctx)

def print_message(message, color=INFO):
    """Helper function to print messages with color"""
    print(f"{color}{message}{RESET}")

# Cache for city coordinates
@lru_cache(maxsize=1000)
def get_city_coordinates(city: str, country: str) -> Optional[Tuple[float, float]]:
    """
    Get latitude and longitude coordinates for a city with caching
    """
    try:
        # Standardize city and country names
        city = CITY_MAPPING.get(city.lower(), city)
        country = COUNTRY_MAPPING.get(country.lower(), country)
        
        location = geolocator.geocode(f"{city}, {country}", timeout=10)
        if location:
            return (location.latitude, location.longitude)
        else:
            print_message(f"Could not find coordinates for {city}, {country}", WARNING)
            return None
    except GeocoderTimedOut:
        print_message(f"Timeout getting coordinates for {city}, {country}", WARNING)
        return None
    except Exception as e:
        print_message(f"Error getting coordinates for {city}, {country}: {str(e)}", ERROR)
        return None

def calculate_distance(departure_city: str, departure_country: str,
                      destination_city: str, destination_country: str) -> float:
    """
    Calculate distance in kilometers between two cities
    """

    if departure_city == destination_city and departure_country == destination_country:
        return 5.0

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
        print_message(f"Could not calculate distance between {departure_city} and {destination_city}", WARNING)
        return 0.0

def extract_city_data(city, date):
    """
    Extract data for a specific city and date
    """
    city_data = {}
    date_str = date.strftime('%Y%m%d')
    
    # Extract personnel data
    personnel_path = f"{BASE_PATH}/BDD_BGES_{city}/PERSONNEL_{city}.txt"
    if os.path.exists(personnel_path):
        personnel_df = pd.read_csv(personnel_path, delimiter=';')
        city_data['personnel'] = personnel_df
    
    # Extract informatique data
    informatique_dir = f"{BASE_PATH}/BDD_BGES_{city}/BDD_BGES_{city}_INFORMATIQUE"
    if os.path.exists(informatique_dir):
        informatique_files = [f for f in os.listdir(informatique_dir) 
                            if f.startswith('MATERIEL_INFORMATIQUE_') and f.endswith(f'{date_str}.txt')]
        if informatique_files:
            df = pd.read_csv(os.path.join(informatique_dir, informatique_files[0]), delimiter=';')
            city_data['informatique'] = df
    
    # Extract mission data
    mission_dir = f"{BASE_PATH}/BDD_BGES_{city}/BDD_BGES_{city}_MISSION"
    if os.path.exists(mission_dir):
        mission_files = [f for f in os.listdir(mission_dir) 
                        if f.startswith('MISSION_') and f.endswith(f'{date_str}.txt')]
        if mission_files:
            df = pd.read_csv(os.path.join(mission_dir, mission_files[0]), delimiter=';')
            city_data['mission'] = df
    
    return city_data

def extract_data_for_date(date):
    """
    Extract data for all cities for a specific date
    """
    city_data_dict = {}
    date_str = date.strftime('%Y-%m-%d')
    
    for city in CITIES:
        city_data = extract_city_data(city, date)
        if city_data:
            city_data_dict[city] = city_data

    return city_data_dict

def transform_mission_data(mission_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform mission data to match fact_business_travel schema
    """
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
        'ALLER_RETOUR': 'IS_ROUND_TRIP',
    })
    
    # Convert date to datetime
    transformed['DATE_ID'] = pd.to_datetime(transformed['DATE_ID'])
    
    # Convert boolean values
    transformed['IS_ROUND_TRIP'] = transformed['IS_ROUND_TRIP'].map({'oui': True, 'non': False})
    
    # Translate mission types to English
    transformed['MISSION_TYPE_ID'] = transformed['MISSION_TYPE_ID'].str.lower().map(MISSION_TYPE_TRANSLATIONS)

    # Translate transport types to English
    transformed['TRANSPORT_ID'] = transformed['TRANSPORT_ID'].str.lower().map(TRANSPORT_TYPE_TRANSLATIONS)
    
    # Calculate distances
    transformed['DISTANCE_KM'] = transformed.apply(
        lambda row: calculate_distance(
            row['DEPARTURE_CITY'],
            row['DEPARTURE_COUNTRY'],
            row['DESTINATION_CITY'],
            row['DESTINATION_COUNTRY']
        ),
        axis=1
    )
    
    # Validate distances
    # Calculate median distances for each transport type
    median_distances = transformed.groupby('TRANSPORT_ID')['DISTANCE_KM'].median()
    
    # Handle invalid distances (negative, null, or unrealistic)
    invalid_count = 0
    
    # Define maximum reasonable distances (in km) for each transport type
    max_distances = {
        'taxi': 100,  # Typically used for short distances
        'public transport': 100,  # Typically used for urban/suburban travel
        'train': 2000,  # Long-distance trains
        'plane': 20000  # Long-haul flights
    }
    
    for transport_type in transformed['TRANSPORT_ID'].unique():
        # Get the median distance for this transport type
        median_distance = median_distances.get(transport_type, 0)
        
        # Find invalid records for this transport type
        invalid_mask = (
            (transformed['TRANSPORT_ID'] == transport_type) & 
            (
                (transformed['DISTANCE_KM'].isna()) |  # Null distances
                (transformed['DISTANCE_KM'] <= 0) |     # Negative and 0 distances
                (transformed['DISTANCE_KM'] > max_distances.get(transport_type, float('inf')))  # Unrealistic distances
            )
        )
        
        invalid_records = transformed[invalid_mask]
        if not invalid_records.empty:
            transformed.loc[invalid_mask, 'DISTANCE_KM'] = median_distance
    
    # Double the distance for round trips
    round_trips = transformed['IS_ROUND_TRIP'].sum()
    transformed.loc[transformed['IS_ROUND_TRIP'], 'DISTANCE_KM'] *= 2
    
    # Select only the columns we need
    required_columns = [
        'TRAVEL_ID', 'EMPLOYEE_ID', 'MISSION_TYPE_ID', 'DEPARTURE_CITY',
        'DEPARTURE_COUNTRY', 'DESTINATION_CITY', 'DESTINATION_COUNTRY',
        'TRANSPORT_ID', 'DATE_ID', 'DISTANCE_KM', 'IS_ROUND_TRIP'
    ]
    transformed = transformed[required_columns]
    
    return transformed

def transform_all_mission_data(extracted_data: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Transform mission data from all cities into a single DataFrame
    """
    mission_dfs = []
    
    for city, city_data in extracted_data.items():
        if 'mission' in city_data:
            transformed_df = transform_mission_data(city_data['mission'])
            mission_dfs.append(transformed_df)
    
    if mission_dfs:
        result = pd.concat(mission_dfs, ignore_index=True)
        return result
    else:
        return pd.DataFrame()

def transform_personnel_data(personnel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform personnel data to match dim_employee schema
    """
    # Rename columns to match database schema
    transformed = personnel_df.rename(columns={
        'ID_PERSONNEL': 'employee_id',
        'NOM_PERSONNEL': 'last_name',
        'PRENOM_PERSONNEL': 'first_name',
        'DT_NAISS': 'birth_date',
        'VILLE_NAISS': 'birth_city',
        'PAYS_NAISS': 'birth_country',
        'NUM_SECU': 'social_security_number',
        'IND_PAYS_NUM_TELP': 'phone_country_code',
        'NUM_TELEPHONE': 'phone_number',
        'NUM_VOIE': 'address_street_number',
        'DSC_VOIE': 'address_street_name',
        'CMPL_VOIE': 'address_complement',
        'CD_POSTAL': 'postal_code',
        'VILLE': 'current_city',
        'PAYS': 'current_country',
        'FONCTION_PERSONNEL': 'sector_name',
        'TS_CREATION_PERSONNEL': 'creation_date',
        'TS_MAJ_PPERSONNEL': 'last_update_date'
    })
    
    # Convert city and country names to lowercase
    transformed['birth_city'] = transformed['birth_city'].str.lower()
    transformed['birth_country'] = transformed['birth_country'].str.lower()
    transformed['current_city'] = transformed['current_city'].str.lower()
    transformed['current_country'] = transformed['current_country'].str.lower()
    
    # Convert date to datetime, if the date is not valid, set it to None
    transformed['birth_date'] = pd.to_datetime(transformed['birth_date'], errors='coerce')
    transformed['creation_date'] = pd.to_datetime(transformed['creation_date'], errors='coerce')
    transformed['last_update_date'] = pd.to_datetime(transformed['last_update_date'], errors='coerce')

    # Need to map the sector_name to the translated sector_name
    transformed['sector_name'] = transformed['sector_name'].map(SECTOR_NAME_TRANSLATIONS)

    return transformed

def transform_all_personnel_data(extracted_data: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Transform personnel data from all cities into a single DataFrame
    """
    personnel_dfs = []
    
    for city, city_data in extracted_data.items():
        if 'personnel' in city_data:
            transformed_df = transform_personnel_data(city_data['personnel'])
            personnel_dfs.append(transformed_df)
    
    if personnel_dfs:
        result = pd.concat(personnel_dfs, ignore_index=True)
        return result
    else:
        return pd.DataFrame()

def transform_equipment_data(equipment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform equipment data to match dim_equipment schema
    """
    # Rename columns to match database schema
    transformed = equipment_df.rename(columns={
        'ID_MATERIELINFO': 'equipment_id',
        'TYPE': 'equipment_type',
        'MODELE': 'model',
        'DATE_ACHAT': 'purchase_date',
        'ID_PERSONNEL': 'employee_id'  # Add employee_id mapping
    })
    
    # We need to get the CO2 impact from materiel_informatique_impact.csv
    impact_file = os.path.join(BASE_PATH, "materiel_informatique_impact.csv")
    if os.path.exists(impact_file):
        impact_df = pd.read_csv(impact_file, delimiter=',')
        
        # Create a mapping of model to equipment type
        model_type_mapping = dict(zip(impact_df['Modèle'], impact_df['Type']))
        
        # Fill in missing equipment types based on model
        missing_types = transformed['equipment_type'].isna() | (transformed['equipment_type'] == ' ')
        if missing_types.any():
            transformed.loc[missing_types, 'equipment_type'] = transformed.loc[missing_types, 'model'].map(model_type_mapping)
        
        # Create a mapping of equipment type to CO2 impact
        impact_mapping = dict(zip(impact_df['Type'], impact_df['Impact']))
        # Map the CO2 impact to each equipment
        transformed['co2_impact_kg'] = transformed['equipment_type'].map(impact_mapping)
    else:
        transformed['co2_impact_kg'] = 0
    
    # Convert CO2 impact to decimal
    transformed['co2_impact_kg'] = pd.to_numeric(transformed['co2_impact_kg'], errors='coerce')
    
    # Convert purchase date to datetime
    transformed['purchase_date'] = pd.to_datetime(transformed['purchase_date'], errors='coerce')
    
    # Select only the columns we need
    required_columns = ['equipment_id', 'equipment_type', 'model', 'co2_impact_kg', 'purchase_date', 'employee_id']
    transformed = transformed[required_columns]
    
    return transformed

def transform_all_equipment_data(extracted_data: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Transform equipment data from all cities into a single DataFrame
    """
    equipment_dfs = []
    
    for city, city_data in extracted_data.items():
        if 'informatique' in city_data:
            transformed_df = transform_equipment_data(city_data['informatique'])
            equipment_dfs.append(transformed_df)
    
    if equipment_dfs:
        result = pd.concat(equipment_dfs, ignore_index=True)
        return result
    else:
        return pd.DataFrame()

def create_db_engine():
    """
    Create a connection to the PostgreSQL database
    """
    try:
        # Create SQLAlchemy engine
        engine = sqlalchemy_create_engine(
            f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
        
        return engine
        
    except Exception as e:
        print_message(f"Error connecting to database: {str(e)}", ERROR)
        raise

def load_dimension_tables(transformed_missions: pd.DataFrame, transformed_equipment: pd.DataFrame):
    """
    Load data into dimension tables from transformed mission and personnel data
    """
    try:
        # Create database engine
        engine = create_db_engine()
        
        with engine.connect() as conn:
            # Load dim_mission_type
            mission_types = transformed_missions['MISSION_TYPE_ID'].unique()
            for i, mission_type in enumerate(mission_types, 1):
                conn.execute(
                    text("""
                        INSERT INTO dim_mission_type (mission_type_id, mission_type_name)
                        VALUES (:mission_type_id, :mission_type_name)
                        ON CONFLICT (mission_type_id) DO NOTHING
                    """),
                    {"mission_type_id": i, "mission_type_name": mission_type}
                )
            conn.commit()

            # Load dim_location
            # First get all unique locations from missions
            mission_locations = pd.concat([
                transformed_missions[['DEPARTURE_CITY', 'DEPARTURE_COUNTRY']].rename(
                    columns={'DEPARTURE_CITY': 'city', 'DEPARTURE_COUNTRY': 'country'}
                ),
                transformed_missions[['DESTINATION_CITY', 'DESTINATION_COUNTRY']].rename(
                    columns={'DESTINATION_CITY': 'city', 'DESTINATION_COUNTRY': 'country'}
                )
            ]).drop_duplicates()

            # Get all unique locations from employees
            employee_locations = pd.read_sql("""
                SELECT DISTINCT current_city as city, current_country as country
                FROM dim_employee
            """, conn)

            # Combine all locations
            all_locations = pd.concat([mission_locations, employee_locations]).drop_duplicates()
            
            # Convert to lowercase for consistency
            all_locations['city'] = all_locations['city'].str.lower()
            all_locations['country'] = all_locations['country'].str.lower()
            
            # Insert all locations
            for i, (_, row) in enumerate(all_locations.iterrows(), 1):
                conn.execute(
                    text("""
                        INSERT INTO dim_location (location_id, city, country)
                        VALUES (:location_id, :city, :country)
                        ON CONFLICT (location_id) DO NOTHING
                    """),
                    {"location_id": i, "city": row['city'], "country": row['country']}
                )
            conn.commit()

            # Load dim_date_time
            mission_dates = transformed_missions['DATE_ID'].unique()
            equipment_dates = transformed_equipment['purchase_date'].unique() if not transformed_equipment.empty else []
            all_dates = pd.to_datetime(pd.concat([pd.Series(mission_dates), pd.Series(equipment_dates)]).unique())
            
            for date in all_dates:
                conn.execute(
                    text("""
                        INSERT INTO dim_date_time (date_id, date, day, month, year, hour, minute, second)
                        VALUES (:date_id, :date, :day, :month, :year, :hour, :minute, :second)
                        ON CONFLICT (date_id) DO NOTHING
                    """),
                    {
                        "date_id": date,
                        "date": date.date(),
                        "day": date.day,
                        "month": date.month,
                        "year": date.year,
                        "hour": date.hour,
                        "minute": date.minute,
                        "second": date.second
                    }
                )
            conn.commit()

            # Load dim_transport
            transport_factors = pd.read_csv('data/transport_factors_by_subcategory.tsv', sep='\t')
            transport_mapping = {
                'taxi': float(transport_factors[transport_factors['subcategory'] == 'taxi']['mean_emissions'].iloc[0]),
                'plane': float(transport_factors[transport_factors['subcategory'] == 'plane']['mean_emissions'].iloc[0]),
                'train': float(transport_factors[transport_factors['subcategory'] == 'train']['mean_emissions'].iloc[0]),
                'public transport': float(transport_factors[transport_factors['subcategory'] == 'public transport']['mean_emissions'].iloc[0])
            }
            
            transport_types = transformed_missions['TRANSPORT_ID'].unique()
            for i, transport in enumerate(transport_types, 1):
                conn.execute(
                    text("""
                        INSERT INTO dim_transport (transport_id, transport_name, emission_factor)
                        VALUES (:transport_id, :transport_name, :emission_factor)
                        ON CONFLICT (transport_id) DO NOTHING
                    """),
                    {"transport_id": i, "transport_name": transport, "emission_factor": transport_mapping.get(transport.lower(), 0.0)}
                )
            conn.commit()

            # Load dim_equipment
            if not transformed_equipment.empty:
                conn.execute(
                    text("""
                        INSERT INTO dim_equipment (equipment_id, equipment_type, model, co2_impact_kg)
                        VALUES (:equipment_id, :equipment_type, :model, :co2_impact_kg)
                        ON CONFLICT (equipment_id) DO NOTHING
                    """),
                    transformed_equipment.to_dict(orient='records')
                )
                conn.commit()

        # Close the engine
        engine.dispose()

    except Exception as e:
        print_message(f"Error loading dimension tables: {str(e)}", ERROR)
        raise

def load_fact_tables(transformed_missions: pd.DataFrame, transformed_equipment: pd.DataFrame):
    """
    Load data into fact tables after dimension tables are populated
    """
    try:
        # Create database engine
        engine = create_db_engine()
        
        with engine.connect() as conn:
            # First, get all the necessary mappings from dimension tables
            # Get location mappings
            locations = pd.read_sql("SELECT location_id, city, country FROM dim_location", conn)
            location_mapping = {(row['city'].lower(), row['country'].lower()): row['location_id'] 
                              for _, row in locations.iterrows()}
            
            # Get transport mappings
            transports = pd.read_sql("SELECT transport_id, transport_name FROM dim_transport", conn)
            transport_mapping = {row['transport_name'].lower(): row['transport_id'] 
                               for _, row in transports.iterrows()}
            
            # Get mission type mappings
            mission_types = pd.read_sql("SELECT mission_type_id, mission_type_name FROM dim_mission_type", conn)
            mission_type_mapping = {row['mission_type_name'].lower(): row['mission_type_id'] 
                                  for _, row in mission_types.iterrows()}
            
            # Load fact_business_travel
            if not transformed_missions.empty:
                # Map locations
                transformed_missions['departure_location_id'] = transformed_missions.apply(
                    lambda row: location_mapping.get((row['DEPARTURE_CITY'].lower(), row['DEPARTURE_COUNTRY'].lower())), 
                    axis=1
                )
                transformed_missions['destination_location_id'] = transformed_missions.apply(
                    lambda row: location_mapping.get((row['DESTINATION_CITY'].lower(), row['DESTINATION_COUNTRY'].lower())), 
                    axis=1
                )
                
                # Map transport and mission types
                transformed_missions['transport_id'] = transformed_missions['TRANSPORT_ID'].str.lower().map(transport_mapping)
                transformed_missions['mission_type_id'] = transformed_missions['MISSION_TYPE_ID'].str.lower().map(mission_type_mapping)
                
                # Insert into fact_business_travel
                conn.execute(
                    text("""
                        INSERT INTO fact_business_travel (
                            travel_id, employee_id, mission_type_id, departure_location_id,
                            destination_location_id, transport_id, date_id, distance_km,
                            is_round_trip
                        )
                        VALUES (
                            :travel_id, :employee_id, :mission_type_id, :departure_location_id,
                            :destination_location_id, :transport_id, :date_id, :distance_km,
                            :is_round_trip
                        )
                        ON CONFLICT (travel_id) DO NOTHING
                    """),
                    transformed_missions.rename(columns={
                        'TRAVEL_ID': 'travel_id',
                        'EMPLOYEE_ID': 'employee_id',
                        'DATE_ID': 'date_id',
                        'DISTANCE_KM': 'distance_km',
                        'IS_ROUND_TRIP': 'is_round_trip'
                    }).to_dict(orient='records')
                )
                conn.commit()
            
            # Load fact_employee_equipment
            if not transformed_equipment.empty:
                # Get employee mappings
                employees = pd.read_sql("""
                    SELECT employee_id, current_city, current_country 
                    FROM dim_employee
                """, conn)
                employee_mapping = set(employees['employee_id'])
                
                # Filter equipment to only those assigned to valid employees
                valid_equipment = transformed_equipment[transformed_equipment['employee_id'].isin(employee_mapping)]
                
                if not valid_equipment.empty:
                    # Create a mapping of employee_id to their location
                    employee_location_mapping = {
                        row['employee_id']: (row['current_city'].lower(), row['current_country'].lower())
                        for _, row in employees.iterrows()
                    }
                    
                    # Add location_id to equipment data and handle missing values
                    valid_equipment['location_id'] = valid_equipment['employee_id'].map(
                        lambda x: location_mapping.get(employee_location_mapping.get(x, (None, None)))
                    )
                    
                    # Convert location_id to integer, replacing NaN with None
                    valid_equipment['location_id'] = pd.to_numeric(valid_equipment['location_id'], errors='coerce')
                    valid_equipment['location_id'] = valid_equipment['location_id'].astype('Int64')  # Use nullable integer type
                    
                    # Prepare the data for insertion
                    equipment_data = valid_equipment.rename(columns={
                        'equipment_id': 'id_materiel',
                        'purchase_date': 'purchase_date_id'
                    }).copy()
                    
                    # Add equipment_id (same as id_materiel)
                    equipment_data['equipment_id'] = equipment_data['id_materiel']
                    
                    # Select only the required columns
                    equipment_data = equipment_data[[
                        'id_materiel', 'equipment_id', 'employee_id', 
                        'location_id', 'purchase_date_id'
                    ]]
                    
                    # Convert to dict and handle None values
                    records = equipment_data.to_dict(orient='records')
                    for record in records:
                        if pd.isna(record['location_id']):
                            record['location_id'] = None
                            print_message(f"Location ID is NaN for equipment {record['id_materiel']}", WARNING)
                    
                    # Insert into fact_employee_equipment
                    conn.execute(
                        text("""
                            INSERT INTO fact_employee_equipment (
                                id_materiel, equipment_id, employee_id, location_id, purchase_date_id
                            )
                            VALUES (
                                :id_materiel, :equipment_id, :employee_id, :location_id, :purchase_date_id
                            )
                            ON CONFLICT (id_materiel) DO NOTHING
                        """),
                        records
                    )
                    conn.commit()
            
        # Close the engine
        engine.dispose()

    except Exception as e:
        print_message(f"Error loading fact tables: {str(e)}", ERROR)
        raise

def extract_personnel_data():
    """
    Extract personnel data for all cities
    """
    personnel_dfs = []
    
    for city in CITIES:
        personnel_path = f"{BASE_PATH}/BDD_BGES_{city}/PERSONNEL_{city}.txt"
        if os.path.exists(personnel_path):
            personnel_df = pd.read_csv(personnel_path, delimiter=';')
            personnel_dfs.append(personnel_df)
    
    if personnel_dfs:
        return pd.concat(personnel_dfs, ignore_index=True)
    else:
        return pd.DataFrame()

def process_date(date):
    """
    Process ETL for a specific date
    """
    date_str = date.strftime('%Y-%m-%d')
    print_message(f"Analyzing {date_str}")
    start_time = time.time()
    
    try:
        # Extract data
        data = extract_data_for_date(date)
        
        # Transform data
        transformed_missions = transform_all_mission_data(data)
        transformed_equipment = transform_all_equipment_data(data)

        # Load data
        load_dimension_tables(transformed_missions, transformed_equipment)
        load_fact_tables(transformed_missions, transformed_equipment)
        
        elapsed_time = time.time() - start_time
        print_message(f"✓ {date_str} ETL completed in {elapsed_time:.2f}s", SUCCESS)
        
    except Exception as e:
        print_message(f"Error processing {date_str}: {str(e)}", ERROR)
        raise

if __name__ == "__main__":
    print_message("Starting ETL process")
    
    # First, process personnel data for all cities
    print_message("Processing personnel data for all cities...")
    personnel_data = extract_personnel_data()
    transformed_personnel = transform_personnel_data(personnel_data)
    
    # Load sector data first
    print_message("Loading sector data into dimension tables...")
    try:
        engine = create_db_engine()
        with engine.connect() as conn:
            # Load dim_sector
            sectors = transformed_personnel['sector_name'].unique()
            for i, sector in enumerate(sectors, 1):
                conn.execute(
                    text("""
                        INSERT INTO dim_sector (sector_id, sector_name)
                        VALUES (:sector_id, :sector_name)
                        ON CONFLICT (sector_id) DO NOTHING
                    """),
                    {"sector_id": i, "sector_name": sector}
                )
            conn.commit()
            
            # Create sector mapping for employee data
            sector_mapping = {sector: i+1 for i, sector in enumerate(sectors)}
            transformed_personnel['sector_id'] = transformed_personnel['sector_name'].map(sector_mapping)
            
        engine.dispose()
        print_message("✓ Sector data loaded successfully", SUCCESS)
    except Exception as e:
        print_message(f"Error loading sector data: {str(e)}", ERROR)
        raise
    
    # Load personnel data into dim_employee
    print_message("Loading personnel data into dimension tables...")
    try:
        engine = create_db_engine()
        with engine.connect() as conn:
            # Load dim_employee
            if not transformed_personnel.empty:
                conn.execute(
                    text("""
                        INSERT INTO dim_employee (
                            employee_id, last_name, first_name, birth_date, birth_city, birth_country,
                            social_security_number, phone_country_code, phone_number, address_street_number,
                            address_street_name, address_complement, postal_code, current_city, current_country,
                            sector_id, creation_date, last_update_date
                        )
                        VALUES (
                            :employee_id, :last_name, :first_name, :birth_date, :birth_city, :birth_country,
                            :social_security_number, :phone_country_code, :phone_number, :address_street_number,
                            :address_street_name, :address_complement, :postal_code, :current_city, :current_country,
                            :sector_id, :creation_date, :last_update_date
                        )
                        ON CONFLICT (employee_id) DO NOTHING
                    """),
                    transformed_personnel.to_dict(orient='records')
                )
                conn.commit()
        engine.dispose()
        print_message("✓ Personnel data loaded successfully", SUCCESS)
    except Exception as e:
        print_message(f"Error loading personnel data: {str(e)}", ERROR)
        raise
    
    # Get list of dates from mission files
    mission_dates = set()
    
    for city in CITIES:
        mission_dir = f"{BASE_PATH}/BDD_BGES_{city}/BDD_BGES_{city}_MISSION"
        if os.path.exists(mission_dir):
            for file in os.listdir(mission_dir):
                if file.startswith('MISSION_') and file.endswith('.txt'):
                    try:
                        date_str = file.split('_')[-1].replace('.txt', '')
                        date = datetime.strptime(date_str, '%Y%m%d')
                        mission_dates.add(date)
                    except ValueError:
                        continue
    
    total_dates = len(mission_dates)
    print_message(f"Found {total_dates} dates to process")
    
    # Process each date
    for i, date in enumerate(sorted(mission_dates), 1):
        process_date(date)
    
    print_message("ETL process completed for all dates", SUCCESS)
