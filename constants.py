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
    'PEKIN': 'beijing',
    'NEW-YORK': 'new york',
    'DUBAI': 'dubai',
    'SIDNEY': 'sydney',
    'MEXICO': 'mexico city',
    'LOSANGELES': 'los angeles',
    'NEWYORK': 'new york'
}

COUNTRY_MAPPING = {
    'allemagne': 'germany',
    'usa': 'united states',
    'emirats': 'united arab emirates',
    'norvège': 'norway',
    'suède': 'sweden',
    'finlande': 'finland',
    'tunisie': 'tunisia',
    'maroc': 'morocco',
    'france': 'france',
    'england': 'united kingdom',
    'japan': 'japan',
    'china': 'china',
    'brazil': 'brazil',
    'canada': 'canada',
    'australia': 'australia',
    'new zealand': 'new zealand',
    'argentina': 'argentina',
    'colombia': 'colombia',
    'peru': 'peru',
    'algeria': 'algeria'
}

SECTOR_NAME_TRANSLATIONS = {
    # German to English
    'Dateningenieur': 'data engineer',
    'Computeringenieur': 'computer engineer',
    'Führungskraft': 'business executive',
    'Ökonom': 'economist',
    'Personalleiter': 'hrd',
    
    # French to English
    'Ingénieur Data': 'data engineer',
    'Ingénieur Informaticien': 'computer engineer',
    'Cadre': 'business executive',
    'Economiste': 'economist',
    'DRH': 'hrd',

    # US to English
    'Data Engineer': 'data engineer',
    'Computer Engineer': 'computer engineer',
    'Business Executive': 'business executive',
    'Economist': 'economist',
    'HRD': 'hrd',
}

TRANSPORT_TYPE_TRANSLATIONS = {
    # French to English
    'avion': 'plane',
    'transports en commun': 'public transport',
    'train': 'train',
    'taxi': 'taxi',
}