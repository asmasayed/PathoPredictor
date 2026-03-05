import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import webbrowser

print("Building PathoPredictor Interactive Web Dashboard...")

# 1. Run the Math Engine (SEIR) for a specific local outbreak (N=5000)
N = 5000 
I0, E0, R0 = 20, 50, 0
S0 = N - I0 - E0 - R0
beta = 0.45  # Example transmission rate adjusted for smaller scale
gamma = 0.1  # Recovery rate
sigma = 0.2  # Incubation rate
days = 60

S, E, I, R = [S0], [E0], [I0], [R0]
for _ in range(days):
    dS = -beta * S[-1] * I[-1] / N
    dE = beta * S[-1] * I[-1] / N - sigma * E[-1]
    dI = sigma * E[-1] - gamma * I[-1]
    dR = gamma * I[-1]
    S.append(S[-1] + dS); E.append(E[-1] + dE); I.append(I[-1] + dI); R.append(R[-1] + dR)

# 2. Build the SEIR Graph Panel
fig_seir = go.Figure()
fig_seir.add_trace(go.Scatter(y=S, mode='lines', name='Susceptible', line=dict(color='blue')))
fig_seir.add_trace(go.Scatter(y=E, mode='lines', name='Exposed', line=dict(color='orange')))
fig_seir.add_trace(go.Scatter(y=I, mode='lines', name='Infectious', line=dict(color='red', width=3)))
fig_seir.add_trace(go.Scatter(y=R, mode='lines', name='Recovered', line=dict(color='green')))
fig_seir.update_layout(title="H5N1 Local Outbreak Forecast Curve", xaxis_title="Days", yaxis_title="Population")
seir_div = fig_seir.to_html(full_html=False, include_plotlyjs='cdn')

# 3. Build the US Geospatial Map Panel
map_div = "<p>Map Data Loading...</p>"
map_path = "data/raw/time_series/h5n1_state_map.csv"
if os.path.exists(map_path):
    df_map = pd.read_csv(map_path)
    if 'State_Code' in df_map.columns:
        fig_map = px.choropleth(df_map, locations='State_Code', locationmode="USA-states", 
                                color='Total_Cases', scope="usa", color_continuous_scale="Reds", 
                                title="Geospatial Hotspots: Active H5N1 Mammal Infections")
        map_div = fig_map.to_html(full_html=False, include_plotlyjs=False)

# 4. Generate Weekly AI Conclusions
conclusions_html = "<div style='background-color:#f4f4f4; padding:20px; border-radius:10px;'><h3>📊 Weekly S-E-I-R Forecast & AI Insights</h3><ul>"
for week in range(1, 9):
    day = week * 7 - 1
    s, e, i, r = S[day], E[day], I[day], R[day]
    
    # Auto-Conclusion Logic
    if week > 1 and I[day] > I[day-7] * 1.5:
        insight = f"<span style='color:red; font-weight:bold;'>⚠️ AI ALERT: Infections surging! Beta transmission rate is accelerating.</span>"
    elif week > 1 and I[day] < I[day-7]:
        insight = f"<span style='color:green; font-weight:bold;'>✅ AI INSIGHT: Outbreak is stabilizing. The peak has been passed.</span>"
    else:
        insight = "<i>AI INSIGHT: Disease is incubating at baseline expected rate.</i>"
        
    conclusions_html += f"<li style='margin-bottom:15px; font-size: 16px;'><b>Week {week} (Day {day+1}):</b> Susceptible: {int(s)} | Exposed: {int(e)} | Infectious: <span style='color:red; font-weight:bold;'>{int(i)}</span> | Recovered: {int(r)} <br> {insight}</li>"
conclusions_html += "</ul></div>"

# 5. Compile the Final HTML Website
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PathoPredictor AI Dashboard</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; }}
        h1 {{ color: #2c3e50; text-align: center; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .container {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
        .panel {{ width: 48%; margin-bottom: 20px; }}
        .full-panel {{ width: 100%; margin-top: 20px; }}
    </style>
</head>
<body>
    <h1>🔬 PathoPredictor: H5N1 Epidemic Intelligence Dashboard</h1>
    <div class="container">
        <div class="panel">{seir_div}</div>
        <div class="panel">{map_div}</div>
    </div>
    <div class="full-panel">{conclusions_html}</div>
</body>
</html>
"""

# Save and Open automatically
with open("dashboard.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Success! Opening Dashboard in your web browser...")
webbrowser.open("dashboard.html")