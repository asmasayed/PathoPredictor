import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import json
import webbrowser

print("Building PathoPredictor Global Intelligence Dashboard...")

# ==========================================
# 1. THE GLOBAL DROPDOWN (Reading the AI Math)
# ==========================================
print(" -> Loading AI-Calculated Forecasts...")
try:
    with open("data/processed/global_seir_forecasts.json", "r") as f:
        global_data = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError("Could not find global_seir_forecasts.json! Please run run_pipeline.py first.")

fig_seir = go.Figure()
regions = list(global_data.keys())
buttons = []

# Loop through the regions and draw the lines
for i, region in enumerate(regions):
    data = global_data[region]
    
    # Only make the first region (US) visible by default
    is_visible = True if i == 0 else False
    
    # Add the 4 S-E-I-R lines for this region
    fig_seir.add_trace(go.Scatter(y=data["S"], mode='lines', name=f'{region} Susceptible', line=dict(color='#3498db'), visible=is_visible))
    fig_seir.add_trace(go.Scatter(y=data["E"], mode='lines', name=f'{region} Exposed', line=dict(color='#f39c12'), visible=is_visible))
    fig_seir.add_trace(go.Scatter(y=data["I"], mode='lines', name=f'{region} Infectious', line=dict(color='#e74c3c', width=3), visible=is_visible))
    fig_seir.add_trace(go.Scatter(y=data["R"], mode='lines', name=f'{region} Recovered', line=dict(color='#2ecc71'), visible=is_visible))

    visibility_array = [False] * (len(regions) * 4)
    for j in range(4):
        visibility_array[(i * 4) + j] = True
        
    buttons.append(
        dict(
            label=f"🌍 {region} Forecast",
            method="update",
            args=[{"visible": visibility_array},
                  {"title": f"PathoPredictor AI Forecast: {region}"}]
        )
    )

fig_seir.update_layout(
    title=f"PathoPredictor AI Forecast: {regions[0]}",
    xaxis_title="Days Forecasted",
    yaxis_title="Population Count",
    plot_bgcolor='white',
    paper_bgcolor='white',
    updatemenus=[dict(active=0, buttons=buttons, x=0.0, xanchor="left", y=1.15, yanchor="top")]
)
fig_seir.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
fig_seir.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
seir_div = fig_seir.to_html(full_html=False, include_plotlyjs='cdn')


# ==========================================
# 2. THE WEEKLY REPORT TABLES (What I missed!)
# ==========================================
print(" -> Generating Numerical Weekly Reports...")
reports_html = "<div class='table-container'>"

for region in regions:
    data = global_data[region]
    table_html = f"<div class='table-card'>"
    table_html += f"<h3>📊 {region} Weekly Math</h3>"
    table_html += "<table>"
    table_html += "<tr><th>Timeline</th><th>Susceptible</th><th>Exposed</th><th>Infectious</th><th>Recovered</th></tr>"
    
    # Slice the 60-day simulation into 7-day jumps
    for day in range(0, 60, 7):
        s_val = int(data["S"][day])
        e_val = int(data["E"][day])
        i_val = int(data["I"][day])
        r_val = int(data["R"][day])
        table_html += f"<tr><td><b>Day {day}</b></td><td>{s_val:,}</td><td>{e_val:,}</td><td style='color:#e74c3c; font-weight:bold;'>{i_val:,}</td><td>{r_val:,}</td></tr>"
        
    table_html += "</table></div>"
    reports_html += table_html
    
reports_html += "</div>"


# ==========================================
# 3. THE TIME-LAPSE GPS MAP
# ==========================================
print(" -> Generating Global Time-Lapse GPS Map...")

map_urls = {
    "US": "https://raw.githubusercontent.com/fbranda/avian-flu/main/Americas/USA/hpai-mammals.csv",
    "AFRICA": "https://raw.githubusercontent.com/fbranda/avian-flu/main/Africa/africa-outbreaks.csv",
    "ASIA": "https://raw.githubusercontent.com/fbranda/avian-flu/main/Asia/asia-outbreaks.csv"
}

map_frames = []

for region, url in map_urls.items():
    try:
        df = pd.read_csv(url)
        date_col = next((col for col in df.columns if 'date' in col.lower()), None)
        lat_col = next((col for col in df.columns if 'lat' in col.lower()), None)
        lon_col = next((col for col in df.columns if 'lon' in col.lower()), None)
        
        if date_col and lat_col and lon_col:
            df['Date'] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True).dt.strftime('%Y-%m')
            df = df.dropna(subset=['Date', lat_col, lon_col])
            geo_df = df.groupby(['Date', lat_col, lon_col]).size().reset_index(name='Cases')
            geo_df.columns = ['Date', 'Latitude', 'Longitude', 'Cases']
            map_frames.append(geo_df)
    except Exception as e:
        print(f"⚠️ Map warning for {region}: {e}")

if len(map_frames) > 0:
    global_map_df = pd.concat(map_frames).sort_values('Date')
    fig_map = px.scatter_geo(
        global_map_df,
        lat="Latitude",
        lon="Longitude",
        size="Cases",
        color="Cases",
        animation_frame="Date",
        projection="natural earth",
        color_continuous_scale="Reds",
        title="Live Global Viral Spread Time-Lapse (Press Play)"
    )
    fig_map.update_traces(marker=dict(sizemin=4))
    fig_map.update_layout(plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=0, r=0, t=40, b=0))
    map_div = fig_map.to_html(full_html=False, include_plotlyjs=False)
else:
    map_div = "<p>Map data could not be rendered.</p>"


# ==========================================
# 4. COMPILE HTML AND LAUNCH
# ==========================================
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PathoPredictor AI Dashboard</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; margin: 0; background-color: #ecf0f1; padding: 20px; }}
        h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 15px; margin-bottom: 30px; }}
        .top-container {{ display: flex; flex-wrap: wrap; justify-content: space-between; gap: 20px; margin-bottom: 20px; }}
        .panel {{ width: 48%; background: white; padding: 15px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); box-sizing: border-box; }}
        
        /* New Table Styles */
        .table-container {{ display: flex; justify-content: space-between; gap: 20px; margin-top: 10px; }}
        .table-card {{ width: 32%; background: white; padding: 15px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); box-sizing: border-box; }}
        .table-card h3 {{ text-align: center; color: #34495e; margin-top: 0; font-size: 16px; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; text-align: right; font-size: 13px; }}
        th {{ text-align: right; padding: 8px; border-bottom: 2px solid #bdc3c7; color: #7f8c8d; }}
        th:first-child, td:first-child {{ text-align: left; }}
        td {{ padding: 8px; border-bottom: 1px solid #ecf0f1; color: #2c3e50; }}
        tr:hover {{ background-color: #f9f9f9; }}
    </style>
</head>
<body>
    <h1>🔬 PathoPredictor: Global Epidemic Intelligence Engine</h1>
    
    <div class="top-container">
        <div class="panel">{seir_div}</div>
        <div class="panel">{map_div}</div>
    </div>
    
    <h2 style="color: #2c3e50; margin-top: 30px; font-size: 20px;">Raw Mathematical Projections (60-Day Horizon)</h2>
    {reports_html}
    
</body>
</html>
"""

with open("dashboard.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("✅ Success! Opening Interactive Dashboard in your web browser...")
webbrowser.open("dashboard.html")