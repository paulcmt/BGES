import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import time
from datetime import datetime
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Database connection parameters
DB_USER = 'postgres'
DB_PASSWORD = 'postgres'
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'postgres'

# Create database engine
def create_db_engine():
    return create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# Function to get table counts
def get_table_counts(engine):
    with engine.connect() as conn:
        tables = ['dim_employee', 'dim_equipment', 'dim_location', 'dim_mission_type', 
                 'dim_sector', 'dim_transport', 'dim_date_time', 'fact_business_travel', 
                 'fact_employee_equipment']
        
        counts = {}
        for table in tables:
            try:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                counts[table] = result.scalar()
            except:
                counts[table] = 0
        return counts

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

# Function to get transport statistics
def get_transport_stats(engine):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT t.transport_name, COUNT(*) as count, 
                   SUM(bt.distance_km) as total_distance,
                   SUM(bt.distance_km * t.emission_factor) as total_emissions
            FROM fact_business_travel bt
            JOIN dim_transport t ON bt.transport_id = t.transport_id
            GROUP BY t.transport_name
        """))
        return pd.DataFrame(result.fetchall(), columns=['Transport', 'Count', 'Total Distance', 'Total Emissions'])

# Function to get equipment statistics
def get_equipment_stats(engine):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT e.equipment_type, COUNT(*) as count,
                   SUM(e.co2_impact_kg) as total_impact
            FROM fact_employee_equipment fee
            JOIN dim_equipment e ON fee.equipment_id = e.equipment_id
            GROUP BY e.equipment_type
        """))
        return pd.DataFrame(result.fetchall(), columns=['Equipment Type', 'Count', 'Total Impact'])

# Function to get mission type statistics
def get_mission_type_stats(engine):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT mt.mission_type_name, COUNT(*) as count
            FROM fact_business_travel bt
            JOIN dim_mission_type mt ON bt.mission_type_id = mt.mission_type_id
            GROUP BY mt.mission_type_name
        """))
        return pd.DataFrame(result.fetchall(), columns=['Mission Type', 'Count'])

# Set up the Streamlit page
st.set_page_config(page_title="BGES Data Warehouse Dashboard", layout="wide")
st.title("BGES Data Warehouse Dashboard")

# Add refresh interval selector
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 60, 5)

# Create a container for the metrics
metrics_container = st.container()

# Create columns for the charts
col1, col2 = st.columns(2)

# Create a container for the latest ETL dates
dates_container = st.container()

# Initialize the database engine
engine = create_db_engine()

# Main dashboard loop
while True:
    with metrics_container:
        st.header("Table Statistics")
        counts = get_table_counts(engine)
        
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
        transport_stats = get_transport_stats(engine)
        
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
        equipment_stats = get_equipment_stats(engine)
        
        # Create a pie chart for equipment counts
        fig = px.pie(equipment_stats, values='Count', names='Equipment Type',
                    title='Distribution of Equipment Types')
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a bar chart for impact
        fig = px.bar(equipment_stats, x='Equipment Type', y='Total Impact',
                    title='Total CO2 Impact by Equipment Type')
        st.plotly_chart(fig, use_container_width=True)
    
    with dates_container:
        st.header("Latest ETL Dates")
        latest_dates = get_latest_etl_dates(engine)
        for date in latest_dates:
            st.write(f"✓ {date.strftime('%Y-%m-%d')}")
        
        # Add mission type statistics
        st.header("Mission Type Distribution")
        mission_stats = get_mission_type_stats(engine)
        fig = px.pie(mission_stats, values='Count', names='Mission Type',
                    title='Distribution of Mission Types')
        st.plotly_chart(fig, use_container_width=True)
    
    # Refresh based on selected interval
    time.sleep(refresh_interval)
    st.rerun() 