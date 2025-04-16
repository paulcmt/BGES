import os
import pandas as pd
from datetime import datetime
import psycopg2
from sqlalchemy import create_engine
from typing import Dict, List, Any

# Ignore timezone for pyarrow
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

# Constants
BASE_PATH = "data"
CITIES = ["BERLIN", "LONDON", "LOSANGELES", "NEWYORK", "PARIS", "SHANGHAI"]

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
    
    return city_data

def extract_data():
    """
    Extract data for all cities
    """
    city_data_dict = {}
    for city in CITIES:
        print(f"Extracting data for {city}...")
        city_data = extract_city_data(city)
        if city_data:
            city_data_dict[city] = city_data
            print(f"Successfully extracted data for {city}")
        else:
            print(f"No data found for {city}")
    
    return city_data_dict

if __name__ == "__main__":
    # Test the extraction
    data = extract_data()
    for city, city_data in data.items():
        print(f"\nData for {city}:")
        for data_type, df in city_data.items():
            print(f"- {data_type}: {len(df)} rows") 
