import pandas as pd
from sqlalchemy import text
from ..etl import create_db_engine
from constants import RESET, INFO, SUCCESS, WARNING, ERROR

def verify_table_data(engine, table_name):
    """
    Verify data quality for a specific table
    """
    try:
        print(f"\n{INFO}Verifying table: {table_name}{RESET}")
        
        # Get table data
        with engine.connect() as conn:
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        
        # Check for empty table
        if len(df) == 0:
            print(f"{WARNING}Table {table_name} is empty{RESET}")
            return False
        
        # Check for NaN/None values
        null_counts = df.isnull().sum()
        if null_counts.any():
            print(f"{WARNING}Found null values in {table_name}:{RESET}")
            for column, count in null_counts[null_counts > 0].items():
                print(f"  {column}: {count} null values")
            return False
        
        # Check for empty strings in string columns
        string_columns = df.select_dtypes(include=['object']).columns
        for column in string_columns:
            empty_strings = (df[column] == '').sum()
            if empty_strings > 0:
                print(f"{WARNING}Found {empty_strings} empty strings in {table_name}.{column}{RESET}")
                return False
        
        # Check for zero values in numeric columns that shouldn't be zero
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        for column in numeric_columns:
            if column not in ['day', 'month', 'year', 'hour', 'minute', 'second']:  # Exclude date components
                zero_values = (df[column] == 0).sum()
                if zero_values > 0:
                    print(f"{WARNING}Found {zero_values} zero values in {table_name}.{column}{RESET}")
                    return False
        
        print(f"{SUCCESS}Table {table_name} passed all checks{RESET}")
        return True
    
    except Exception as e:
        print(f"{ERROR}Error verifying table {table_name}: {str(e)}{RESET}")
        return False

def verify_foreign_keys(engine):
    """
    Verify referential integrity of foreign keys
    """
    try:
        print(f"\n{INFO}Verifying foreign key relationships{RESET}")
        
        # Define foreign key relationships with correct column names
        fk_relationships = [
            ('fact_business_travel', 'employee_id', 'dim_employee', 'employee_id'),
            ('fact_business_travel', 'mission_type_id', 'dim_mission_type', 'mission_type_id'),
            ('fact_business_travel', 'departure_location_id', 'dim_location', 'location_id'),
            ('fact_business_travel', 'destination_location_id', 'dim_location', 'location_id'),
            ('fact_business_travel', 'transport_id', 'dim_transport', 'transport_id'),
            ('fact_business_travel', 'date_id', 'dim_date_time', 'date_id'),
            ('fact_employee_equipment', 'equipment_id', 'dim_equipment', 'equipment_id'),
            ('fact_employee_equipment', 'employee_id', 'dim_employee', 'employee_id'),
            ('fact_employee_equipment', 'location_id', 'dim_location', 'location_id'),
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
                    print(f"{WARNING}Found {result} orphaned foreign keys in {table}.{fk_column} referencing {ref_table}.{ref_column}{RESET}")
                    all_valid = False
        
        if all_valid:
            print(f"{SUCCESS}All foreign key relationships are valid{RESET}")
        return all_valid
    
    except Exception as e:
        print(f"{ERROR}Error verifying foreign keys: {str(e)}{RESET}")
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
        
        # Print final result
        if all_tables_valid and fk_valid:
            print(f"\n{SUCCESS}All data quality checks passed successfully{RESET}")
        else:
            print(f"\n{WARNING}Some data quality issues were found. Please review the warnings above.{RESET}")
        
        # Close the engine
        engine.dispose()
        
    except Exception as e:
        print(f"{ERROR}Error during verification: {str(e)}{RESET}")

if __name__ == "__main__":
    main() 