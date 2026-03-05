import pandas as pd
import os

print("Fetching official US H5N1 Outbreak Data...")
url = "https://raw.githubusercontent.com/fbranda/avian-flu/main/Americas/USA/hpai-mammals.csv"
df = pd.read_csv(url)

# 1. Save Time Series for the AI Brain
date_col = next(col for col in df.columns if 'date' in col.lower())
df['Date'] = pd.to_datetime(df[date_col])
clean_df = df.groupby('Date').size().reset_index(name='Cases')

os.makedirs("data/raw/time_series", exist_ok=True)
clean_df.to_csv("data/raw/time_series/h5n1_us_outbreaks.csv", index=False)

# 2. Extract Geospatial Data for the Dashboard Map
state_col = next((col for col in df.columns if 'state' in col.lower()), None)
if state_col:
    state_df = df.groupby(state_col).size().reset_index(name='Total_Cases')
    state_df.columns = ['State', 'Total_Cases']
    
    # AI Map tools require 2-letter state codes, so we translate them here
    state_map = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
        'Colorado': 'CO', 'Florida': 'FL', 'Georgia': 'GA', 'Idaho': 'ID', 'Illinois': 'IL', 
        'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Maryland': 'MD',
        'Michigan': 'MI', 'Minnesota': 'MN', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 
        'Nevada': 'NV', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC',
        'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
        'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 
        'Utah': 'UT', 'Virginia': 'VA', 'Washington': 'WA', 'Wisconsin': 'WI', 'Wyoming': 'WY'
    }
    state_df['State_Code'] = state_df['State'].map(state_map)
    state_df = state_df.dropna(subset=['State_Code'])
    state_df.to_csv("data/raw/time_series/h5n1_state_map.csv", index=False)

print("Success! Downloaded Time Series AND Geospatial Map Data.")