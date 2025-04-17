# ANSI color codes
SUCCESS = '\033[92m'  # Green
INFO = '\033[94m'     # Blue
WARNING = '\033[93m'  # Yellow
ERROR = '\033[91m'    # Red
RESET = '\033[0m'     # Reset

# Data paths
BASE_PATH = "data"
CITIES = ["BERLIN", "LONDON", "LOSANGELES", "NEWYORK", "PARIS", "SHANGHAI"]

# Mission type translations
MISSION_TYPE_TRANSLATIONS = {
    # French to English
    'formation': 'training',
    'conférence': 'conference',
    'réunion': 'meeting',
    'rencontre entreprises': 'business meeting',
    'développement': 'development',
    
    # German to English
    'schulung': 'training',
    'konferenz': 'conference',
    'meeting': 'meeting',
    'geschäftstreffen': 'business meeting',
    'entwicklung': 'development',
    
    # US to English
    'vocational training': 'training', 
    'conference': 'conference', 
    'team meeting': 'meeting',
    'business meeting': 'business meeting',
    'development': 'development', 
}

# City and country name mappings
CITY_MAPPING = {
    'PEKIN': 'Beijing',
    'NEW-YORK': 'New York',
    'DUBAI': 'Dubai',
    'SIDNEY': 'Sydney',
    'MEXICO': 'Mexico City',
    'LOSANGELES': 'Los Angeles',
    'NEWYORK': 'New York'
}

COUNTRY_MAPPING = {
    'Allemagne': 'Germany',
    'USA': 'United States',
    'Emirats': 'United Arab Emirates',
    'Norvège': 'Norway',
    'Suède': 'Sweden',
    'Finlande': 'Finland',
    'Tunisie': 'Tunisia',
    'Maroc': 'Morocco',
    'France': 'France',
    'England': 'United Kingdom',
    'Japan': 'Japan',
    'China': 'China',
    'Brazil': 'Brazil',
    'Canada': 'Canada',
    'Australia': 'Australia',
    'New Zealand': 'New Zealand',
    'Argentina': 'Argentina',
    'Colombia': 'Colombia',
    'Peru': 'Peru',
    'Algeria': 'Algeria'
}

SECTOR_NAME_TRANSLATIONS = {
    # German to English
    'Dateningenieur': 'Data Engineer',
    'Computeringenieur': 'Computer Engineer',
    'Führungskraft': 'Business Executive',
    'Ökonom': 'Economist',
    'Personalleiter': 'HRD',
    
    # French to English
    'Ingénieur Data': 'Data Engineer',
    'Ingénieur Informaticien': 'Computer Engineer',
    'Cadre': 'Business Executive',
    'Economiste': 'Economist',
    'DRH': 'HRD',

    # US to English
    'Data Engineer': 'Data Engineer',
    'Computer Engineer': 'Computer Engineer',
    'Business Executive': 'Business Executive',
    'Economist': 'Economist',
    'HRD': 'HRD',
}