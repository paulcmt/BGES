import pandas as pd

def calculate_transport_factors():
    # Read the transport factors file
    df = pd.read_csv('data/transports_factors.tsv', sep='\t')
    
    # Translate subcategories to English
    translation_map = {
        'Voiture': 'taxi',
        'Avion': 'plane',
        'Ferroviaire': 'train',
        'Bus': 'public transport'
    }
    
    # Define multiplication factors for each transport type (taken from apps.labos1point5.org/documentation)
    multiplication_factors = {
        'taxi': 2.6,  # Round trip and 1.3
        'plane': 1,
        'train': 1.2,
        'public transport': 1.6 # Mean of 1.5 and 1.7
    }
    
    # Apply translation
    df['subcategory'] = df['subcategory'].map(translation_map)
    
    # Calculate means by subcategory
    subcategory_means = df.groupby('subcategory').agg({
        'total': ['mean', 'count']
    }).reset_index()
    
    # Rename columns
    subcategory_means.columns = ['subcategory', 'mean_emissions', 'count']
    
    # Apply multiplication factors
    subcategory_means['mean_emissions'] = subcategory_means.apply(
        lambda row: row['mean_emissions'] * multiplication_factors[row['subcategory']], 
        axis=1
    )
    
    # Round mean_emissions to 4 decimal places
    subcategory_means['mean_emissions'] = subcategory_means['mean_emissions'].round(4)
    
    # Sort by subcategory
    subcategory_means = subcategory_means.sort_values('subcategory')
    
    # Print results
    print("\nMean Transport Emission Factors by Type (kg CO2e per km):")
    print("=" * 80)
    
    for _, row in subcategory_means.iterrows():
        print(f"{row['subcategory']}:")
        print(f"  Mean Emission Factor: {row['mean_emissions']:.4f} kg CO2e/km")
        print(f"  Number of variants: {row['count']}")
        print(f"  Multiplication factor: {multiplication_factors[row['subcategory']]}")
        print("-" * 40)
    
    # Save results to file
    subcategory_means.to_csv('data/transport_factors_by_subcategory.tsv', sep='\t', index=False)
    print("\nResults saved to data/transport_factors_by_subcategory.tsv")

if __name__ == "__main__":
    calculate_transport_factors() 