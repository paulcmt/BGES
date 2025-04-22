# WHAT IS THIS PROJECT ABOUT ?

This project is about creating a Data Warehouse for a company that wants to estimate its GHG emissions.

## HOW TO RUN THE PROJECT ?

1. Clone the repository
2. Create a virtual environment
   - `python -m venv venv`
   - `source venv/bin/activate`
3. Install the dependencies
   - `pip install -r requirements.txt`
4. Run the docker container
   - `docker-compose up -d`
5. Run the streamlit app
   - `streamlit run dashboard.py`
6. Run the ETL process
   - `python etl.py`
