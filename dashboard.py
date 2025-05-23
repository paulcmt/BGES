import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import time
from datetime import datetime, timezone, timedelta
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
from dotenv import load_dotenv

# Load environment variables from db.env
load_dotenv('db.env')

# Database connection parameters from environment variables
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')

# Create database engine
def create_db_engine():
    return create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# Function to get available date range
def get_date_range(engine):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT MIN(date) as min_date, MAX(date) as max_date
            FROM dim_date_time
        """))
        row = result.fetchone()
        return row[0], row[1]

# Function to get table counts with date filter
def get_table_counts(engine, start_date=None, end_date=None):
    with engine.connect() as conn:
        tables = ['dim_employee', 'dim_equipment', 'dim_location', 'dim_mission_type', 
                 'dim_sector', 'dim_transport', 'dim_date_time', 'fact_business_travel', 
                 'fact_employee_equipment']
        
        counts = {}
        for table in tables:
            try:
                if table in ['fact_business_travel', 'fact_employee_equipment'] and start_date is not None and end_date is not None:
                    if table == 'fact_business_travel':
                        query = f"""
                            SELECT COUNT(*) 
                            FROM {table} bt
                            JOIN dim_date_time dt ON bt.date_id = dt.date_id
                            WHERE dt.date BETWEEN :start_date AND :end_date
                        """
                    else:  # fact_employee_equipment
                        query = f"""
                            SELECT COUNT(*) 
                            FROM {table} ee
                            JOIN dim_date_time dt ON ee.purchase_date_id = dt.date_id
                            WHERE dt.date BETWEEN :start_date AND :end_date
                        """
                    result = conn.execute(text(query), {"start_date": start_date, "end_date": end_date})
                else:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                counts[table] = result.scalar()
            except:
                counts[table] = 0
        return counts

# Function to get transport statistics with date filter
def get_transport_stats(engine, start_date=None, end_date=None):
    with engine.connect() as conn:
        query = """
            SELECT t.transport_name, COUNT(*) as count, 
                   SUM(bt.distance_km) as total_distance,
                   SUM(bt.distance_km * t.emission_factor) as total_emissions
            FROM fact_business_travel bt
            JOIN dim_transport t ON bt.transport_id = t.transport_id
            JOIN dim_date_time dt ON bt.date_id = dt.date_id
        """
        if start_date is not None and end_date is not None:
            query += " WHERE dt.date BETWEEN :start_date AND :end_date"
        query += " GROUP BY t.transport_name"
        
        result = conn.execute(text(query), {"start_date": start_date, "end_date": end_date})
        return pd.DataFrame(result.fetchall(), columns=['Transport', 'Count', 'Total Distance', 'Total Emissions'])

# Function to get equipment statistics with date filter
def get_equipment_stats(engine, start_date=None, end_date=None):
    with engine.connect() as conn:
        query = """
            SELECT e.equipment_type, COUNT(*) as count,
                   SUM(e.co2_impact_kg) as total_impact
            FROM fact_employee_equipment fee
            JOIN dim_equipment e ON fee.equipment_id = e.equipment_id
            JOIN dim_date_time dt ON fee.purchase_date_id = dt.date_id
        """
        if start_date is not None and end_date is not None:
            query += " WHERE dt.date BETWEEN :start_date AND :end_date"
        query += " GROUP BY e.equipment_type"
        
        result = conn.execute(text(query), {"start_date": start_date, "end_date": end_date})
        return pd.DataFrame(result.fetchall(), columns=['Equipment Type', 'Count', 'Total Impact'])

# Function to get mission type statistics with date filter
def get_mission_type_stats(engine, start_date=None, end_date=None):
    with engine.connect() as conn:
        query = """
            SELECT mt.mission_type_name, COUNT(*) as count
            FROM fact_business_travel bt
            JOIN dim_mission_type mt ON bt.mission_type_id = mt.mission_type_id
            JOIN dim_date_time dt ON bt.date_id = dt.date_id
        """
        if start_date is not None and end_date is not None:
            query += " WHERE dt.date BETWEEN :start_date AND :end_date"
        query += " GROUP BY mt.mission_type_name"
        
        result = conn.execute(text(query), {"start_date": start_date, "end_date": end_date})
        return pd.DataFrame(result.fetchall(), columns=['Mission Type', 'Count'])

# Function to get latest ETL dates
def get_latest_etl_dates(engine):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT DISTINCT date_trunc('day', date_id) as date
            FROM dim_date_time
            ORDER BY date DESC
            LIMIT 5
        """))
        return [row[0] for row in result]

# Function to check if ETL is still running
def is_etl_running(engine):
    with engine.connect() as conn:
        try:
            # Check if any tables are being modified
            result = conn.execute(text("""
                SELECT 
                    schemaname,
                    relname as table_name,
                    n_live_tup as live_rows,
                    n_dead_tup as dead_rows,
                    last_vacuum,
                    last_autovacuum,
                    last_analyze,
                    last_autoanalyze
                FROM pg_stat_user_tables
                WHERE schemaname = 'public'
                AND relname IN (
                    'dim_employee',
                    'fact_business_travel',
                    'fact_employee_equipment'
                )
            """))
            
            # Get the current state of the tables
            table_status = {row[1]: {
                'live_rows': row[2],
                'dead_rows': row[3],
                'last_vacuum': row[4],
                'last_autovacuum': row[5],
                'last_analyze': row[6],
                'last_autoanalyze': row[7]
            } for row in result}
            
            # If any required table doesn't exist, ETL is still running
            if not all(table in table_status for table in ['dim_employee', 'fact_business_travel', 'fact_employee_equipment']):
                return True
            
            # Check for recent vacuum or analyze operations (within last minute)
            now = datetime.now(timezone.utc)  # Make now timezone-aware
            for table in table_status:
                status = table_status[table]
                for timestamp_field in ['last_vacuum', 'last_autovacuum', 'last_analyze', 'last_autoanalyze']:
                    if status[timestamp_field] is not None:
                        # Convert PostgreSQL timestamp to timezone-aware datetime if needed
                        if status[timestamp_field].tzinfo is None:
                            status[timestamp_field] = status[timestamp_field].replace(tzinfo=timezone.utc)
                        time_diff = (now - status[timestamp_field]).total_seconds()
                        if time_diff < 20:  # 20 seconds
                            return True
            
            # Check for dead rows (indicates recent modifications)
            for table in table_status:
                if table_status[table]['dead_rows'] > 0:
                    return True
            
            # If we get here, no recent activity was detected
            return False
            
        except Exception as e:
            # If there's any error, consider ETL still running
            print(f"Error checking ETL status: {e}")
            return True

# Set up the Streamlit page
st.set_page_config(page_title="BGES Data Warehouse Dashboard", layout="wide")
st.title("BGES Data Warehouse Dashboard")

# Initialize the database engine
engine = create_db_engine()

# Get available date range
min_date, max_date = get_date_range(engine)

# Add date range filters in the sidebar
st.sidebar.header("Date Range Filter")

# Only show date inputs if we have dates in the database
if min_date is not None and max_date is not None:
    start_date = st.sidebar.date_input(
        "Start Date",
        min_date,
        min_value=min_date,
        max_value=max_date
    )
    end_date = st.sidebar.date_input(
        "End Date",
        max_date,
        min_value=min_date,
        max_value=max_date
    )

    # Validate date range
    if start_date > end_date:
        st.sidebar.error("Error: End date must be after start date.")
        st.stop()
else:
    st.sidebar.info("No date data available in the database.")
    start_date = None
    end_date = None

# Add refresh interval selector (minimum 0.1 seconds)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 0.1, 60.0, 0.1, step=0.1)

# Add manual refresh button
if st.sidebar.button("🔄 Manual Refresh"):
    st.rerun()

# Create a container for the metrics
metrics_container = st.container()

# Create columns for the charts
col1, col2 = st.columns(2)

# Create a container for the latest ETL dates
dates_container = st.container()

# Initialize session state for tracking updates
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'is_etl_complete' not in st.session_state:
    st.session_state.is_etl_complete = False
if 'last_counts' not in st.session_state:
    st.session_state.last_counts = None

# Main dashboard display
def display_dashboard():
    with metrics_container:
        st.header("Table Statistics")
        counts = get_table_counts(engine, start_date, end_date)
        
        # Check if counts have changed since last update
        if st.session_state.last_counts is not None and counts == st.session_state.last_counts:
            if not st.session_state.is_etl_complete:
                st.session_state.is_etl_complete = True
                st.sidebar.success("ETL process completed! Dashboard will no longer refresh.")
        else:
            st.session_state.last_counts = counts
        
        # Create metrics in a grid
        cols = st.columns(3)
        for i, (table, count) in enumerate(counts.items()):
            with cols[i % 3]:
                st.metric(
                    label=table.replace('_', ' ').title(),
                    value=count
                )
    
    with col1:
        st.header("Transport Statistics")
        transport_stats = get_transport_stats(engine, start_date, end_date)
        
        # Create a pie chart for transport counts
        fig = px.pie(transport_stats, values='Count', names='Transport', 
                    title='Distribution of Transport Types')
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a bar chart for emissions
        fig = px.bar(transport_stats, x='Transport', y='Total Emissions',
                    title='Total Emissions by Transport Type')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.header("Equipment Statistics")
        equipment_stats = get_equipment_stats(engine, start_date, end_date)
        
        # Create a pie chart for equipment counts
        fig = px.pie(equipment_stats, values='Count', names='Equipment Type',
                    title='Distribution of Equipment Types')
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a bar chart for impact
        fig = px.bar(equipment_stats, x='Equipment Type', y='Total Impact',
                    title='Total CO2 Impact by Equipment Type')
        st.plotly_chart(fig, use_container_width=True)
    
    with dates_container:
        st.header("Selected Date Range")
        if start_date is not None and end_date is not None:
            st.write(f"From: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        else:
            st.write("No date range selected - showing all data")
        
        # Add mission type statistics
        st.header("Mission Type Distribution")
        mission_stats = get_mission_type_stats(engine, start_date, end_date)
        fig = px.pie(mission_stats, values='Count', names='Mission Type',
                    title='Distribution of Mission Types')
        st.plotly_chart(fig, use_container_width=True)
    
    # Check if ETL is still running
    if not st.session_state.is_etl_complete:
        if not is_etl_running(engine):
            st.session_state.is_etl_complete = True
            st.sidebar.success("ETL process completed! Dashboard will no longer refresh.")

# Display the dashboard
display_dashboard()

# Auto-refresh based on selected interval
time.sleep(refresh_interval)
st.rerun()

# If ETL is not complete, set up auto-refresh
if not st.session_state.is_etl_complete:
    time.sleep(refresh_interval)
    st.rerun()
else:
    # Display completion message
    st.sidebar.info("Dashboard is now static. To refresh, please restart the dashboard.") 