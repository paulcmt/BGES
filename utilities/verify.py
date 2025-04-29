import os
import pandas as pd
from sqlalchemy import text
from dotenv import load_dotenv
from etl import create_db_engine
from constants import RESET, INFO, SUCCESS, WARNING, ERROR

# Load environment variables
load_dotenv('db.env')

def print_message(message, color="\033[94m"):
    """Helper function to print colored messages"""
    print(f"{color}{message}\033[0m")

def verify_table_data(engine, table_name):
    """
    Verify data quality for a specific table
    """
    try:
        print(f"\n\033[94mVerifying table: {table_name}\033[0m")
        
        # Get table data
        with engine.connect() as conn:
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        
        # Check for empty table
        if len(df) == 0:
            print(f"\033[93mWarning: Table {table_name} is empty\033[0m")
            return False
        
        # Check for NaN/None values
        null_counts = df.isnull().sum()
        if null_counts.any():
            print(f"\033[93mWarning: Found null values in {table_name}:\033[0m")
            for column, count in null_counts[null_counts > 0].items():
                print(f"  {column}: {count} null values")
            return False
        
        # Check for empty strings in string columns
        string_columns = df.select_dtypes(include=['object']).columns
        for column in string_columns:
            empty_strings = (df[column] == '').sum()
            if empty_strings > 0:
                print(f"\033[93mWarning: Found {empty_strings} empty strings in {table_name}.{column}\033[0m")
                return False
        
        # Check for zero values in numeric columns that shouldn't be zero
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        for column in numeric_columns:
            if column not in ['day', 'month', 'year', 'hour', 'minute', 'second']:  # Exclude date components
                zero_values = (df[column] == 0).sum()
                if zero_values > 0:
                    print(f"\033[93mWarning: Found {zero_values} zero values in {table_name}.{column}\033[0m")
                    return False
        
        print(f"\033[92m✓ Table {table_name} passed all checks\033[0m")
        return True
    
    except Exception as e:
        print(f"\033[91mError verifying table {table_name}: {str(e)}\033[0m")
        return False

def verify_foreign_keys(engine):
    """
    Verify referential integrity of foreign keys
    """
    try:
        print("\n\033[94mVerifying foreign key relationships\033[0m")
        
        # Define foreign key relationships
        fk_relationships = [
            ('fact_business_travel', 'employee_id', 'dim_employee', 'employee_id'),
            ('fact_business_travel', 'mission_type_id', 'dim_mission_type', 'mission_type_id'),
            ('fact_business_travel', 'departure_location_id', 'dim_location', 'location_id'),
            ('fact_business_travel', 'destination_location_id', 'dim_location', 'location_id'),
            ('fact_business_travel', 'transport_id', 'dim_transport', 'transport_id'),
            ('fact_business_travel', 'date_id', 'dim_date_time', 'date_id'),
            ('fact_employee_equipment', 'equipment_id', 'dim_equipment', 'equipment_id'),
            ('fact_employee_equipment', 'employee_id', 'dim_employee', 'employee_id'),
            ('fact_employee_equipment', 'purchase_date_id', 'dim_date_time', 'date_id'),
            ('dim_employee', 'sector_id', 'dim_sector', 'sector_id')
        ]
        
        all_valid = True
        for table, fk_column, ref_table, ref_column in fk_relationships:
            with engine.connect() as conn:
                # Check for orphaned foreign keys
                query = f"""
                    SELECT COUNT(*) 
                    FROM {table} t 
                    LEFT JOIN {ref_table} r ON t.{fk_column} = r.{ref_column}
                    WHERE r.{ref_column} IS NULL
                """
                result = conn.execute(text(query)).scalar()
                
                if result > 0:
                    print(f"\033[93mWarning: Found {result} orphaned foreign keys in {table}.{fk_column} referencing {ref_table}.{ref_column}\033[0m")
                    all_valid = False
        
        if all_valid:
            print("\033[92m✓ All foreign key relationships are valid\033[0m")
        return all_valid
    
    except Exception as e:
        print(f"\033[91mError verifying foreign keys: {str(e)}\033[0m")
        return False

def verify_equipment_data(engine):
    """
    Verify equipment data quality
    """
    try:
        print("\n\033[94mVerifying equipment data\033[0m")
        
        with engine.connect() as conn:
            # Check for equipment types and models that are not strings
            query = """
                SELECT COUNT(*) 
                FROM dim_equipment 
                WHERE equipment_type IS NULL 
                   OR model IS NULL 
                   OR equipment_type = '' 
                   OR model = ''
            """
            result = conn.execute(text(query)).scalar()
            
            if result > 0:
                print(f"\033[93mWarning: Found {result} invalid equipment records (null or empty values)\033[0m")
                return False
            
            # Check for duplicate equipment type and model combinations
            query = """
                SELECT equipment_type, model, COUNT(*) as count
                FROM dim_equipment
                GROUP BY equipment_type, model
                HAVING COUNT(*) > 1
            """
            duplicates = pd.read_sql(query, conn)
            
            if not duplicates.empty:
                print("\033[93mWarning: Found duplicate equipment type and model combinations:\033[0m")
                print(duplicates)
                return False
        
        print("\033[92m✓ Equipment data passed all checks\033[0m")
        return True
    
    except Exception as e:
        print(f"\033[91mError verifying equipment data: {str(e)}\033[0m")
        return False

def main():
    """
    Main verification function
    """
    try:
        # Create database engine
        engine = create_db_engine()
        
        # List of tables to verify
        tables = [
            'dim_sector',
            'dim_location',
            'dim_date_time',
            'dim_employee',
            'dim_equipment',
            'dim_mission_type',
            'dim_transport',
            'fact_business_travel',
            'fact_employee_equipment'
        ]
        
        # Verify each table
        all_tables_valid = True
        for table in tables:
            if not verify_table_data(engine, table):
                all_tables_valid = False
        
        # Verify foreign keys
        fk_valid = verify_foreign_keys(engine)
        
        # Verify equipment data specifically
        equipment_valid = verify_equipment_data(engine)
        
        # Print final result
        if all_tables_valid and fk_valid and equipment_valid:
            print("\n\033[92m✓ All data quality checks passed successfully\033[0m")
        else:
            print("\n\033[93mWarning: Some data quality issues were found. Please review the warnings above.\033[0m")
        
        # Close the engine
        engine.dispose()
        
    except Exception as e:
        print(f"\033[91mError during verification: {str(e)}\033[0m")

if __name__ == "__main__":
    main() 