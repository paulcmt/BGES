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
    
    # Apply translation
    df['subcategory'] = df['subcategory'].map(translation_map)
    
    # Calculate means by subcategory
    subcategory_means = df.groupby('subcategory').agg({
        'total': ['mean', 'count']
    }).reset_index()
    
    # Rename columns
    subcategory_means.columns = ['subcategory', 'mean_emissions', 'count']
    
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
        print("-" * 40)
    
    # Save results to file
    subcategory_means.to_csv('data/transport_factors_by_subcategory.tsv', sep='\t', index=False)
    print("\nResults saved to data/transport_factors_by_subcategory.tsv")

if __name__ == "__main__":
    calculate_transport_factors() 