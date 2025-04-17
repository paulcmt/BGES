from sqlalchemy import create_engine, text
import os

# Database connection parameters
DB_USER = 'postgres'
DB_PASSWORD = 'postgres'
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'postgres'

# Create SQLAlchemy engine
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

def clear_all_tables():
    try:
        with engine.connect() as conn:
            # Disable foreign key checks temporarily
            conn.execute(text("SET session_replication_role = 'replica';"))
            
            # Get all table names
            tables = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """))
            
            # Truncate each table
            for table in tables:
                table_name = table[0]
                print(f"Truncating table: {table_name}")
                conn.execute(text(f"TRUNCATE TABLE {table_name} CASCADE;"))
            
            # Re-enable foreign key checks
            conn.execute(text("SET session_replication_role = 'origin';"))
            
            # Commit the transaction
            conn.commit()
            
            print("✓ Successfully cleared all tables")
            
    except Exception as e:
        print(f"Error clearing tables: {e}")
        raise

if __name__ == "__main__":
    clear_all_tables()
