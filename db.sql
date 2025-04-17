-- Dimension Tables
CREATE TABLE IF NOT EXISTS dim_sector (
    sector_id INT PRIMARY KEY,
    sector_name VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS dim_location (
    location_id INT PRIMARY KEY,
    city VARCHAR(100),
    country VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS dim_date_time (
    date_id TIMESTAMP PRIMARY KEY,
    date DATE,
    day INT,
    month INT,
    year INT,
    hour INT,
    minute INT,
    second INT
);

CREATE TABLE IF NOT EXISTS dim_employee (
    employee_id VARCHAR(50) PRIMARY KEY,
    last_name VARCHAR(100),
    first_name VARCHAR(100),
    birth_date DATE,
    birth_city VARCHAR(100),
    birth_country VARCHAR(100),
    social_security_number VARCHAR(50),
    phone_country_code VARCHAR(10),
    phone_number VARCHAR(50),
    address_street_number VARCHAR(20),
    address_street_name VARCHAR(200),
    address_complement VARCHAR(200),
    postal_code VARCHAR(20),
    current_city VARCHAR(100),
    current_country VARCHAR(100),
    sector_id INT,
    creation_date TIMESTAMP,
    last_update_date TIMESTAMP,
    FOREIGN KEY (sector_id) REFERENCES dim_sector(sector_id)
);

CREATE TABLE IF NOT EXISTS dim_equipment (
    equipment_id VARCHAR(50) PRIMARY KEY,
    equipment_type VARCHAR(100),
    model VARCHAR(100),
    co2_impact_kg DECIMAL(10,2)
);

CREATE TABLE IF NOT EXISTS dim_mission_type (
    mission_type_id INT PRIMARY KEY,
    mission_type_name VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS dim_transport (
    transport_id INT PRIMARY KEY,
    transport_name VARCHAR(100),
    emission_factor DECIMAL(10,4)
);

-- Fact Tables
CREATE TABLE IF NOT EXISTS fact_employee_equipment (
    id_materiel VARCHAR(50) PRIMARY KEY,
    equipment_id VARCHAR(50),
    employee_id VARCHAR(50),
    location_id INT,
    purchase_date_id TIMESTAMP,
    FOREIGN KEY (equipment_id) REFERENCES dim_equipment(equipment_id),
    FOREIGN KEY (employee_id) REFERENCES dim_employee(employee_id),
    FOREIGN KEY (location_id) REFERENCES dim_location(location_id),
    FOREIGN KEY (purchase_date_id) REFERENCES dim_date_time(date_id)
);

CREATE TABLE IF NOT EXISTS fact_business_travel (
    travel_id VARCHAR(50) PRIMARY KEY,
    employee_id VARCHAR(50),
    mission_type_id INT,
    departure_location_id INT,
    destination_location_id INT,
    transport_id INT,
    date_id TIMESTAMP,
    distance_km DECIMAL(10,2),
    is_round_trip BOOLEAN,
    FOREIGN KEY (employee_id) REFERENCES dim_employee(employee_id),
    FOREIGN KEY (mission_type_id) REFERENCES dim_mission_type(mission_type_id),
    FOREIGN KEY (departure_location_id) REFERENCES dim_location(location_id),
    FOREIGN KEY (destination_location_id) REFERENCES dim_location(location_id),
    FOREIGN KEY (transport_id) REFERENCES dim_transport(transport_id),
    FOREIGN KEY (date_id) REFERENCES dim_date_time(date_id)
);

-- Verify tables were created
\dt