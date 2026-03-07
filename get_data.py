import pandas as pd
import os

print("Fetching official Global H5N1 Outbreak Data...")
os.makedirs("data/raw/time_series", exist_ok=True)

# 1. The "Hunter" URLs. We swapped Europe for ASIA.
datasets = {
    "us": ["https://raw.githubusercontent.com/fbranda/avian-flu/main/Americas/USA/hpai-mammals.csv"],
    "africa": ["https://raw.githubusercontent.com/fbranda/avian-flu/main/Africa/africa-outbreaks.csv"],
    "asia": [
        "https://raw.githubusercontent.com/fbranda/avian-flu/main/Asia/asia-outbreaks.csv",
        "https://raw.githubusercontent.com/fbranda/avian-flu/main/Asia/asia-cases.csv"
    ]
}

for region, urls in datasets.items():
    print(f"\nDownloading data for {region.upper()}...")
    df = None
    
    # 2. Try all possible URLs for this region
    for url in urls:
        try:
            df = pd.read_csv(url)
            print(f" -> Successfully found file at: {url}")
            break # Stop searching if we found it!
        except Exception as e:
            print(f" -> [Failed URL] {url} | Error: {e}")
            
    # 3. If we completely failed to find the file, skip to the next region
    if df is None:
        print(f"❌ CRITICAL: Could not download {region.upper()} data from any known URL.")
        continue

    # 4. Clean the data
    try:
        # Find the date column dynamically
        date_col = next((col for col in df.columns if 'date' in col.lower()), df.columns[0])
        
        # Convert to datetime
        df['Date'] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
        df = df.dropna(subset=['Date'])
        
        if len(df) == 0:
            raise ValueError("Dataframe is empty after dropping missing dates. The Date column might be formatted incorrectly.")
            
        clean_df = df.groupby('Date').size().reset_index(name='Cases')
        
        # Save the file safely
        file_path = f"data/raw/time_series/h5n1_{region}_outbreaks.csv"
        clean_df.to_csv(file_path, index=False)
        print(f"✅ Saved {len(clean_df)} daily timeline records to {file_path}")
        
        # 5. Protect the USA Map
        if region == "us":
            state_col = next((col for col in df.columns if 'state' in col.lower()), None)
            if state_col:
                state_df = df.groupby(state_col).size().reset_index(name='Total_Cases')
                state_df.columns = ['State', 'Total_Cases']
                
                state_map = {
                    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
                    'Colorado': 'CO', 'Florida': 'FL', 'Georgia': 'GA', 'Idaho': 'ID', 'Illinois': 'IL', 
                    'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Maryland': 'MD'
                }
                state_df['State_Code'] = state_df['State'].map(state_map)
                state_df = state_df.dropna(subset=['State_Code'])
                state_df.to_csv("data/raw/time_series/h5n1_state_map.csv", index=False)
                print(" -> Saved USA Geospatial Map Data.")
                
    except Exception as e:
        print(f"⚠️ Error cleaning {region.upper()} data: {e}")

print("\n✅ Data Fetching Routine Complete.")