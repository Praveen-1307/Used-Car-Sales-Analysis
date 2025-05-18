import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

# Load the dataset
df = pd.read_csv('../Dataset/dataset.csv')

# Preprocessing for display
# Calculate car age
current_year = 2023
df['Car_Age'] = current_year - df['Year']

# Extract manufacturer from Name column
df['Manufacturer'] = df['Name'].str.split().str[0]

# KPI calculations
total_cars = len(df)
avg_price = df['Price'].mean()
median_age = df['Car_Age'].median()
unique_manu = df['Manufacturer'].nunique()

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True, title="Used Car Price Explorer")

# Sidebar filters
def sidebar():
    return html.Div([
        html.Div([
            html.Div(className="filter-accent"),
            html.H2("Filters", className="sidebar-title")
        ], className="filter-title-row"),
        html.Label("Manufacturer", className="filter-label"),
        dcc.Dropdown(
            id='manufacturer-filter',
            options=[{"label": m, "value": m} for m in sorted(df['Manufacturer'].unique())],
            multi=True,
            placeholder="Select manufacturer(s)",
            className="filter-dropdown"
        ),
        html.Label("Fuel Type", className="filter-label"),
        dcc.Dropdown(
            id='fuel-filter',
            options=[{"label": f, "value": f} for f in sorted(df['Fuel_Type'].unique())],
            multi=True,
            placeholder="Select fuel type(s)",
            className="filter-dropdown"
        ),
        html.Label("Owner Type", className="filter-label"),
        dcc.Dropdown(
            id='owner-filter',
            options=[{"label": o, "value": o} for o in sorted(df['Owner_Type'].unique())],
            multi=True,
            placeholder="Select owner type(s)",
            className="filter-dropdown"
        ),
        html.Label("Price Range (Lakhs ₹)", className="filter-label"),
        dcc.RangeSlider(
            id='price-filter',
            min=df['Price'].min(),
            max=df['Price'].max(),
            value=[df['Price'].min(), df['Price'].max()],
            marks={int(df['Price'].min()): f"₹{int(df['Price'].min())}L", int(df['Price'].max()): f"₹{int(df['Price'].max())}L"},
            step=0.5,
            className="filter-slider"
        ),
        html.Label("Car Age (Years)", className="filter-label"),
        dcc.RangeSlider(
            id='age-filter',
            min=df['Car_Age'].min(),
            max=df['Car_Age'].max(),
            value=[df['Car_Age'].min(), df['Car_Age'].max()],
            marks={int(df['Car_Age'].min()): str(int(df['Car_Age'].min())), int(df['Car_Age'].max()): str(int(df['Car_Age'].max()))},
            step=1,
            className="filter-slider"
        ),
        html.Button("Reset Filters", id="reset-btn", n_clicks=0, className="reset-btn-modern"),
        html.Br(),
        html.Br(),
        html.P("Data Source: CarDekho", className="sidebar-footer")
    ], className="sidebar-modern")

# KPI cards
def kpi_cards(filtered_df):
    return html.Div([
        html.Div([
            html.H4("Total Cars"),
            html.H2(f"{len(filtered_df):,}")
        ], className="kpi-card"),
        html.Div([
            html.H4("Avg. Price (Lakh ₹)"),
            html.H2(f"{filtered_df['Price'].mean():.2f}")
        ], className="kpi-card"),
        html.Div([
            html.H4("Median Age (yrs)"),
            html.H2(f"{filtered_df['Car_Age'].median():.1f}")
        ], className="kpi-card"),
        html.Div([
            html.H4("Manufacturers"),
            html.H2(f"{filtered_df['Manufacturer'].nunique()}")
        ], className="kpi-card")
    ], className="kpi-row")

# Layout
app.layout = html.Div([
    html.Div([
        html.Div("Used Car Price Explorer", className="topbar-title"),
        html.Div([
            dcc.Tabs(id="tabs", value="overview", children=[
                dcc.Tab(label="Overview", value="overview", className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(label="Price Distribution", value="price-dist", className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(label="Manufacturer Insights", value="manu-insights", className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(label="Location Map", value="map", className="custom-tab", selected_className="custom-tab--selected"),
            ], className="tabs")
        ], className="tab-container")
    ], className="topbar"),
    html.Div([
        sidebar(),
        html.Div(id="main-content", className="main-content")
    ], className="main-row")
], className="main-app")

# Callbacks for filtering and tab content
@app.callback(
    Output('main-content', 'children'),
    [Input('tabs', 'value'),
     Input('manufacturer-filter', 'value'),
     Input('fuel-filter', 'value'),
     Input('owner-filter', 'value'),
     Input('price-filter', 'value'),
     Input('age-filter', 'value'),
     Input('reset-btn', 'n_clicks')],
    [State('main-content', 'children')]
)
def update_content(tab, manu, fuel, owner, price, age, reset, prev):
    # Filtering
    filtered = df.copy()
    if manu: filtered = filtered[filtered['Manufacturer'].isin(manu)]
    if fuel: filtered = filtered[filtered['Fuel_Type'].isin(fuel)]
    if owner: filtered = filtered[filtered['Owner_Type'].isin(owner)]
    if price: filtered = filtered[(filtered['Price'] >= price[0]) & (filtered['Price'] <= price[1])]
    if age: filtered = filtered[(filtered['Car_Age'] >= age[0]) & (filtered['Car_Age'] <= age[1])]

    # Overview Tab
    if tab == "overview":
        return html.Div([
            kpi_cards(filtered),
            dcc.Graph(
                figure=px.line(filtered.groupby('Year')['Price'].mean().reset_index(), x='Year', y='Price',
                                title="Average Price Trend by Year", markers=True)
            ),
            dcc.Graph(
                figure=px.bar(
                    filtered['Manufacturer'].value_counts().head(10).reset_index(),
                    x='Manufacturer', y='count',
                    title="Top 10 Manufacturers by Listings",
                    labels={'Manufacturer': 'Manufacturer', 'count': 'Count'}
                )
            )
        ])
    # Price Distribution Tab
    elif tab == "price-dist":
        return html.Div([
            dcc.Graph(
                figure=px.histogram(filtered, x='Price', nbins=30, color='Fuel_Type',
                                    title="Price Distribution by Fuel Type", barmode='overlay')
            ),
            dcc.Graph(
                figure=px.box(filtered, x='Fuel_Type', y='Price', color='Fuel_Type',
                              title="Boxplot: Price by Fuel Type")
            ),
            dcc.Graph(
                figure=px.scatter(filtered, x='Car_Age', y='Price', color='Manufacturer',
                                  title="Price vs Car Age", opacity=0.6)
            )
        ])
    # Manufacturer Insights Tab
    elif tab == "manu-insights":
        return html.Div([
            dcc.Graph(
                figure=px.bar(filtered.groupby('Manufacturer')['Price'].mean().sort_values(ascending=False).head(15).reset_index(),
                              x='Manufacturer', y='Price', color='Price',
                              title="Top 15 Manufacturers by Avg. Price")
            ),
            dcc.Graph(
                figure=px.box(filtered, x='Manufacturer', y='Price', color='Manufacturer',
                              title="Boxplot: Price by Manufacturer", points='all')
            )
        ])
    # Location Map Tab
    elif tab == "map":
        if 'Location' in filtered.columns:
            map_df = filtered.groupby('Location')['Price'].mean().reset_index()
            return html.Div([
                dcc.Graph(
                    figure=px.choropleth_mapbox(
                        map_df, geojson=None, locations='Location', color='Price',
                        center={"lat": 22.9734, "lon": 78.6569}, zoom=3.5,
                        mapbox_style="carto-positron", title="Avg. Price by City (approximate)")
                )
            ])
        else:
            return html.Div("Location data not available.")
    return html.Div("No content.")

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>Used Car Price Explorer</title>
    {%favicon%}
    {%css%}
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f7f9fa; margin: 0; }
        .main-app { min-height: 100vh; }
        /* Topbar and tabs */
        .topbar {
            background: linear-gradient(90deg, #2c3e50 0%, #3498db 100%);
            box-shadow: 0 2px 8px #e0e0e0;
            padding: 0 30px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            height: 70px;
        }
        .topbar-title {
            font-size: 2.2em;
            font-weight: 700;
            color: #fff;
            flex: 1;
            padding: 0 0 0 10px;
            letter-spacing: 1px;
            text-shadow: 1px 1px 6px #000;
        }
        .tab-container {
            display: flex;
            align-items: center;
            height: 100%;
        }
        .tabs {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            margin-left: 30px;
        }
        .custom-tab {
            background: transparent !important;
            border-radius: 18px 18px 0 0 !important;
            color: #fff !important;
            font-size: 1.1em !important;
            font-weight: 500 !important;
            margin: 0 2px !important;
            padding: 12px 28px !important;
            border: none !important;
            transition: background 0.2s, color 0.2s;
        }
        .custom-tab:hover {
            background: rgba(52, 152, 219, 0.10) !important;
            color: #fff !important;
        }
        .custom-tab--selected {
            background: #2c3e50 !important;
            color: #fff !important;
            font-weight: 700 !important;
            box-shadow: 0 2px 8px #e0e0e0;
            border-bottom: 3px solid #3498db !important;
        }
        .main-row { display: flex; }
        /* Modern Sidebar */
        .sidebar-modern {
            width: 300px;
            background: #fff;
            border-radius: 18px;
            box-shadow: 0 4px 24px #e0e0e0, 0 1.5px 4px #3498db33;
            padding: 32px 28px 24px 28px;
            margin: 30px 0 30px 0;
            display: flex;
            flex-direction: column;
            align-items: stretch;
        }
        .filter-title-row {
            display: flex;
            align-items: center;
            margin-bottom: 18px;
        }
        .filter-accent {
            width: 7px;
            height: 32px;
            background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
            border-radius: 6px;
            margin-right: 14px;
        }
        .sidebar-title {
            color: #2c3e50;
            font-size: 1.5em;
            font-weight: 700;
            margin: 0;
            letter-spacing: 0.5px;
        }
        .filter-label {
            font-weight: 600;
            margin-top: 18px;
            margin-bottom: 7px;
            color: #2c3e50;
            font-size: 1.08em;
        }
        .filter-dropdown {
            margin-bottom: 6px;
            font-size: 1.08em;
            border-radius: 8px !important;
            background: #f7f9fa !important;
            color: #2c3e50 !important;
            box-shadow: 0 1px 4px #e0e0e0;
            border: 1px solid #3498db33 !important;
        }
        .filter-slider {
            margin: 10px 0 0 0;
            padding: 0 2px;
        }
        .reset-btn-modern {
            margin-top: 28px;
            background: linear-gradient(90deg, #2c3e50 0%, #3498db 100%);
            color: #fff;
            border: none;
            border-radius: 8px;
            padding: 12px 0;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 2px 8px #3498db55;
            transition: background 0.2s, box-shadow 0.2s;
        }
        .reset-btn-modern:hover {
            background: linear-gradient(90deg, #3498db 0%, #2c3e50 100%);
            color: #fff;
            box-shadow: 0 4px 16px #3498db99;
        }
        .sidebar-footer {
            color: #2c3e5099;
            font-size: 0.98em;
            margin-top: 40px;
            text-align: left;
        }
        .main-content { flex: 1; padding: 30px 40px; }
        .kpi-row { display: flex; gap: 30px; margin-bottom: 30px; }
        .kpi-card { background: #fff; border-radius: 12px; box-shadow: 0 2px 8px #e0e0e0; padding: 22px 30px; flex: 1; text-align: center; border-top: 4px solid #2c3e50; }
        .kpi-card h4 { color: #2c3e50; font-size: 1.1em; margin-bottom: 8px; }
        .kpi-card h2 { font-size: 2.1em; margin: 0; color: #3498db; text-shadow: 1px 1px 4px #3498db33; }
        @media (max-width: 1100px) {
            .main-row { flex-direction: column; }
            .sidebar-modern { width: 100%; border-radius: 0; box-shadow: none; margin: 0 0 18px 0; }
            .main-content { padding: 20px 5vw; }
            .kpi-row { flex-direction: column; gap: 18px; }
            .topbar { flex-direction: column; height: auto; padding: 18px 10px; }
            .tab-container { margin-top: 10px; }
        }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
'''

if __name__ == '__main__':
    app.run_server(debug=True) 