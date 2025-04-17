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
        city = CITY_MAPPING.get(city.lower(), city)
        country = COUNTRY_MAPPING.get(country.lower(), country)
        
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
            print(f"{SUCCESS}✓ Successfully extracted informatique data for {city} from {len(informatique_files)} files{RESET}")
    
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
            print(f"{SUCCESS}✓ Successfully extracted mission data for {city} from {len(mission_files)} files{RESET}")
    
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
        'ALLER_RETOUR': 'IS_ROUND_TRIP',
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

    # Translate transport types to English
    transformed['TRANSPORT_ID'] = transformed['TRANSPORT_ID'].str.lower().map(TRANSPORT_TYPE_TRANSLATIONS)
    print(f"{SUCCESS}✓ Transport type translation completed{RESET}")
    
    # Calculate distances
    print(f"{INFO}Calculating distances between cities...{RESET}")
    # transformed['DISTANCE_KM'] = transformed.apply(
    #     lambda row: calculate_distance(
    #         row['DEPARTURE_CITY'],
    #         row['DEPARTURE_COUNTRY'],
    #         row['DESTINATION_CITY'],
    #         row['DESTINATION_COUNTRY']
    #     ),
    #     axis=1
    # )
    transformed['DISTANCE_KM'] = 0

    print(f"{SUCCESS}✓ Calculated distances for {len(transformed)} missions{RESET}")
    
    # Double the distance for round trips
    round_trips = transformed['IS_ROUND_TRIP'].sum()
    transformed.loc[transformed['IS_ROUND_TRIP'], 'DISTANCE_KM'] *= 2
    print(f"{SUCCESS}✓ Adjusted distances for {round_trips} round trips{RESET}")
    
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

def transform_personnel_data(personnel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform personnel data to match dim_employee schema
    """
    print(f"{INFO}Starting personnel data transformation...{RESET}")
    
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

    print(f"{SUCCESS}✓ Personnel data transformation completed{RESET}")
    return transformed

def transform_all_personnel_data(extracted_data: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Transform personnel data from all cities into a single DataFrame
    """
    personnel_dfs = []
    print(f"{INFO}Starting transformation of all personnel data...{RESET}")
    
    for city, city_data in extracted_data.items():
        if 'personnel' in city_data:
            print(f"{INFO}Transforming personnel data for {city}...{RESET}")
            transformed_df = transform_personnel_data(city_data['personnel'])
            personnel_dfs.append(transformed_df)
            print(f"{SUCCESS}✓ Successfully transformed personnel data for {city}{RESET}")
    
    if personnel_dfs:
        result = pd.concat(personnel_dfs, ignore_index=True)
        print(f"{SUCCESS}✓ All personnel data transformation completed successfully{RESET}")
        return result
    else:
        print(f"{WARNING}No personnel data found to transform{RESET}")
        return pd.DataFrame()

def transform_equipment_data(equipment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform equipment data to match dim_equipment schema
    """
    print(f"{INFO}Starting equipment data transformation...{RESET}")
    
    # Rename columns to match database schema
    transformed = equipment_df.rename(columns={
        'ID_MATERIELINFO': 'equipment_id',
        'TYPE': 'equipment_type',
        'MODELE': 'model',
        'DATE_ACHAT': 'purchase_date',
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
            print(f"{INFO}Filling in {missing_types.sum()} missing equipment types based on model{RESET}")
            transformed.loc[missing_types, 'equipment_type'] = transformed.loc[missing_types, 'model'].map(model_type_mapping)
        
        # Create a mapping of equipment type to CO2 impact
        impact_mapping = dict(zip(impact_df['Type'], impact_df['Impact']))
        # Map the CO2 impact to each equipment
        transformed['co2_impact_kg'] = transformed['equipment_type'].map(impact_mapping)
    else:
        print(f"{WARNING}Equipment impact file not found, setting CO2 impact to 0{RESET}")
        transformed['co2_impact_kg'] = 0
    
    # Convert CO2 impact to decimal
    transformed['co2_impact_kg'] = pd.to_numeric(transformed['co2_impact_kg'], errors='coerce')

    # Convert purchase date to datetime
    transformed['purchase_date'] = pd.to_datetime(transformed['purchase_date'], errors='coerce')
    
    # Select only the columns we need
    required_columns = ['equipment_id', 'equipment_type', 'model', 'co2_impact_kg', 'purchase_date']
    transformed = transformed[required_columns]
    
    # Print statistics about filled equipment types
    filled_count = missing_types.sum() if 'missing_types' in locals() else 0
    if filled_count > 0:
        print(f"{SUCCESS}✓ Filled in {filled_count} missing equipment types{RESET}")
    
    print(f"{SUCCESS}✓ Equipment data transformation completed{RESET}")
    return transformed

def transform_all_equipment_data(extracted_data: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Transform equipment data from all cities into a single DataFrame
    """
    equipment_dfs = []
    print(f"{INFO}Starting transformation of all equipment data...{RESET}")
    
    for city, city_data in extracted_data.items():
        if 'informatique' in city_data:
            print(f"{INFO}Transforming equipment data for {city}...{RESET}")
            transformed_df = transform_equipment_data(city_data['informatique'])
            equipment_dfs.append(transformed_df)
            print(f"{SUCCESS}✓ Successfully transformed equipment data for {city}{RESET}")
    
    if equipment_dfs:
        result = pd.concat(equipment_dfs, ignore_index=True)
        print(f"{SUCCESS}✓ All equipment data transformation completed successfully{RESET}")
        return result
    else:
        print(f"{WARNING}No equipment data found to transform{RESET}")
        return pd.DataFrame()

def create_db_engine():
    """
    Create a connection to the PostgreSQL database
    """
    try:
        # Connection parameters
        db_params = {
            'dbname': 'postgres',
            'user': 'postgres',
            'password': 'postgres',
            'host': 'localhost',
            'port': '5432'
        }
        
        # Create SQLAlchemy engine
        engine = sqlalchemy_create_engine(
            f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
        )
        
        print(f"{SUCCESS}✓ Successfully connected to the database{RESET}")
        return engine
        
    except Exception as e:
        print(f"{ERROR}Error connecting to database: {str(e)}{RESET}")
        raise

def load_dimension_tables(transformed_missions: pd.DataFrame, transformed_personnel: pd.DataFrame, transformed_equipment: pd.DataFrame):
    """
    Load data into dimension tables from transformed mission and personnel data
    """
    print(f"{INFO}Starting dimension tables loading...{RESET}")
    
    try:
        # Create database engine
        engine = create_db_engine()
        
        with engine.connect() as conn:
            # Load dim_mission_type
            mission_types = transformed_missions['MISSION_TYPE_ID'].unique()
            
            i = 1
            for mission_type in mission_types:
                conn.execute(
                    text("""
                        INSERT INTO dim_mission_type (mission_type_id, mission_type_name)
                        VALUES (:mission_type_id, :mission_type_name)
                        ON CONFLICT (mission_type_id) DO NOTHING
                    """),
                    {"mission_type_id": i, "mission_type_name": mission_type}
                )
                i += 1
            conn.commit()
            print(f"{SUCCESS}✓ Loaded {len(mission_types)} mission types{RESET}")

            # Load dim_location
            locations = pd.concat([
                transformed_missions[['DEPARTURE_CITY', 'DEPARTURE_COUNTRY']].rename(
                    columns={'DEPARTURE_CITY': 'city', 'DEPARTURE_COUNTRY': 'country'}
                ),
                transformed_missions[['DESTINATION_CITY', 'DESTINATION_COUNTRY']].rename(
                    columns={'DESTINATION_CITY': 'city', 'DESTINATION_COUNTRY': 'country'}
                ),
                transformed_personnel[['birth_city', 'birth_country']].rename(
                    columns={'birth_city': 'city', 'birth_country': 'country'}
                ),
                transformed_personnel[['current_city', 'current_country']].rename(
                    columns={'current_city': 'city', 'current_country': 'country'}
                )
            ]).drop_duplicates()

            locations['city'] = locations['city'].str.lower()
            locations['country'] = locations['country'].str.lower()
            
            # Create a dictionary with location_id as key and location data as value
            locations_dict = {i+1: row for i, (_, row) in enumerate(locations.iterrows())}
            
            for location_id, row in locations_dict.items():
                conn.execute(
                    text("""
                        INSERT INTO dim_location (location_id, city, country)
                        VALUES (:location_id, :city, :country)
                        ON CONFLICT (location_id) DO NOTHING
                    """),
                    {"location_id": location_id, "city": row['city'], "country": row['country']}
                )
            conn.commit()
            print(f"{SUCCESS}✓ Loaded {len(locations)} locations{RESET}")

            # Load dim_date_time
            # Get all unique timestamps from mission dates and equipment purchase dates
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
            print(f"{SUCCESS}✓ Loaded {len(all_dates)} dates{RESET}")

            # Load dim_transport
            transport_types = transformed_missions['TRANSPORT_ID'].unique()
            
            # Create a dictionary with transport_id as key and transport data as value
            transport_types_dict = {i+1: {'transport_name': transport} 
                                  for i, transport in enumerate(transport_types)}
            
            for transport_id, row in transport_types_dict.items():
                conn.execute(
                    text("""
                        INSERT INTO dim_transport (transport_id, transport_name, emission_factor)
                        VALUES (:transport_id, :transport_name, :emission_factor)
                        ON CONFLICT (transport_id) DO NOTHING
                    """),
                    {"transport_id": transport_id, "transport_name": row['transport_name'], "emission_factor": 0}
                )
            conn.commit()
            print(f"{SUCCESS}✓ Loaded {len(transport_types)} transport types with emission factors{RESET}")

            # Load dim_sector
            sectors = transformed_personnel['sector_name'].unique()

            # Create a dictionary with sector_id as key and sector data as value
            sectors_dict = {i+1: {'sector_name': sector} 
                          for i, sector in enumerate(sectors)}
            
            for sector_id, row in sectors_dict.items():
                conn.execute(
                    text("""
                        INSERT INTO dim_sector (sector_id, sector_name)
                        VALUES (:sector_id, :sector_name)
                        ON CONFLICT (sector_id) DO NOTHING
                    """),
                    {"sector_id": sector_id, "sector_name": row['sector_name']}
                )
            conn.commit()
            print(f"{SUCCESS}✓ Loaded {len(sectors)} sectors{RESET}")

            # Load dim_employee
            if transformed_personnel.empty:
                print(f"{WARNING}No personnel data found to load{RESET}")
            else:
                # Map sector IDs
                sector_mapping = {row['sector_name']: sector_id for sector_id, row in sectors_dict.items()}
                transformed_personnel['sector_id'] = transformed_personnel['sector_name'].map(sector_mapping)

                conn.execute(
                    text("""
                        INSERT INTO dim_employee (employee_id, last_name, first_name, birth_date, birth_city, birth_country, 
                                                social_security_number, phone_country_code, phone_number, address_street_number, 
                                                address_street_name, address_complement, postal_code, current_city, current_country, 
                                                sector_id, creation_date, last_update_date)
                        VALUES (:employee_id, :last_name, :first_name, :birth_date, :birth_city, :birth_country, 
                                :social_security_number, :phone_country_code, :phone_number, :address_street_number, 
                                :address_street_name, :address_complement, :postal_code, :current_city, :current_country, 
                                :sector_id, :creation_date, :last_update_date)
                        ON CONFLICT (employee_id) DO NOTHING
                    """),
                    transformed_personnel.to_dict(orient='records')
                )
                conn.commit()
                print(f"{SUCCESS}✓ Loaded {len(transformed_personnel)} employees{RESET}")

            # Load dim_equipment
            if transformed_equipment.empty:
                print(f"{WARNING}No equipment data found to load{RESET}")
            else:
                conn.execute(
                    text("""
                        INSERT INTO dim_equipment (equipment_id, equipment_type, model, co2_impact_kg)
                        VALUES (:equipment_id, :equipment_type, :model, :co2_impact_kg)
                        ON CONFLICT (equipment_id) DO NOTHING
                    """),
                    transformed_equipment.to_dict(orient='records')
                )
                conn.commit()
                print(f"{SUCCESS}✓ Loaded {len(transformed_equipment)} equipment items{RESET}")

        # Close the engine
        engine.dispose()
        print(f"{SUCCESS}✓ Database connection closed{RESET}")

    except Exception as e:
        print(f"{ERROR}Error loading dimension tables: {str(e)}{RESET}")
        raise

if __name__ == "__main__":
    print(f"{INFO}Starting ETL process...{RESET}")
    
    print("\nExtracting data...")
    data = extract_data()
    
    print("\nTransforming mission data...")
    transformed_missions = transform_all_mission_data(data)

    print("\nTransforming personnel data...")
    transformed_personnel = transform_all_personnel_data(data)

    print("\nTransforming equipment data...")
    transformed_equipment = transform_all_equipment_data(data)

    print("\nLoading dimension tables...")
    load_dimension_tables(transformed_missions, transformed_personnel, transformed_equipment)
    
    print(f"\n{SUCCESS}ETL process completed successfully{RESET}")
