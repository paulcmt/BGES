# WHAT IS THIS PROJECT ABOUT ?

This project is about creating a Data Warehouse for a company that wants to estimate its GHG emissions.

## HOW TO RUN THE PROJECT ?

1. Clone the repository
2. Install docker (cf https://www.docker.com/products/docker-desktop/)
3. Create a virtual environment
   - `python -m venv venv`
   - on Mac: `source venv/bin/activate`
   - on Windows: `source venv/Scripts/activate`
4. Install the dependencies
   - `pip install -r requirements.txt`
5. Run the docker container
   - `docker-compose up -d`
6. Add the data to the database
   - `docker exec -i bges-postgres-1 psql -U postgres -d postgres < db.sql`
7. Run the ETL process
   - `python etl.py`
8. Run the streamlit app to visualize the data filled in the database in real time
   - `streamlit run dashboard.py`
