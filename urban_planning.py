import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw, MarkerCluster
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import rasterio
from rasterio.io import MemoryFile
# import geopandas as gpd
# from shapely.geometry import Polygon
import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.cluster import KMeans
# import time

import os
from dotenv import load_dotenv

load_dotenv()

# Set page configuration
st.set_page_config(layout="wide", page_title="Urban Planning Map Tool - Pakistan")


# Securely store API key
OPENTOPOGRAPHY_API_KEY = os.getenv("API_KEY")

# Define all available parameters
ALL_PARAMETERS = {
    "T2M": "Temperature at 2 Meters (°C)",
    "T2M_MAX": "Maximum Temperature at 2 Meters (°C)",
    "T2M_MIN": "Minimum Temperature at 2 Meters (°C)",
    "PRECTOTCORR": "Precipitation (mm/day)",
    "RH2M": "Relative Humidity at 2 Meters (%)",
    "ALLSKY_SFC_SW_DWN": "All Sky Surface Shortwave Downward Irradiance (W/m^2)",
    "WS2M": "Wind Speed at 2 Meters (m/s)",
    "T2MDEW": "Dew/Frost Point at 2 Meters (°C)",
    "ALLSKY_SFC_LW_DWN": "All Sky Surface Longwave Downward Irradiance (W/m^2)",
    "CLOUD_AMT": "Cloud Amount (%)",
    "GWETROOT": "Root Zone Soil Wetness (%)",
    "QV2M": "Specific Humidity at 2 Meters (kg/kg)",
    "PS": "Surface Pressure (kPa)",
    "T2MWET": "Wet Bulb Temperature at 2 Meters (°C)",
    "ALLSKY_SFC_PAR_TOT": "All Sky Surface Photosynthetically Active Radiation (W/m^2)"
}

@st.cache_data(ttl=3600)
def get_nasa_power_data(lat, lon, start_date, end_date, selected_params):
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": ",".join(selected_params),
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON"
    }
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data['properties']['parameter'])
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.replace(-999, np.nan)
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch NASA data: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_opentopography_data(south, north, west, east):
    dataset = 'SRTMGL1'
    output_format = 'GTiff'
    url = f'https://portal.opentopography.org/API/globaldem?demtype={dataset}&south={south}&north={north}&west={west}&east={east}&outputFormat={output_format}&API_Key={OPENTOPOGRAPHY_API_KEY}'
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with MemoryFile(response.content) as memfile:
            with memfile.open() as dataset:
                elevation_data = dataset.read(1)
                return elevation_data
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch OpenTopography data: {str(e)}")
        return None

def calculate_centroid(coordinates):
    lat_sum = sum(coord[1] for coord in coordinates[0])
    lon_sum = sum(coord[0] for coord in coordinates[0])
    count = len(coordinates[0])
    return lat_sum / count, lon_sum / count

def create_climate_visualizations(nasa_data):
    st.subheader("Climate Data Visualizations")

    plot_data = {
        "Temperature": ["T2M", "T2M_MAX", "T2M_MIN"],
        "Precipitation": ["PRECTOTCORR"],
        "Relative Humidity": ["RH2M"],
        "Solar Radiation": ["ALLSKY_SFC_SW_DWN"]
    }

    available_plots = [key for key, params in plot_data.items() if any(param in nasa_data.columns for param in params)]

    if not available_plots:
        st.warning("No data available for visualization. Please select more parameters.")
        return

    fig = make_subplots(rows=2, cols=2, subplot_titles=available_plots[:4])

    for i, plot_type in enumerate(available_plots[:4], 1):
        row = (i - 1) // 2 + 1
        col = (i - 1) % 2 + 1

        for param in plot_data[plot_type]:
            if param in nasa_data.columns:
                if plot_type == "Precipitation":
                    fig.add_trace(go.Bar(x=nasa_data.index, y=nasa_data[param], name=ALL_PARAMETERS[param]), row=row, col=col)
                else:
                    fig.add_trace(go.Scatter(x=nasa_data.index, y=nasa_data[param], name=ALL_PARAMETERS[param]), row=row, col=col)

    fig.update_layout(height=800, width=1000, title_text=f"Climate Data ({nasa_data.index.min().date()} to {nasa_data.index.max().date()})")
    fig.update_xaxes(title_text="Date")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Additional Climate Parameters")
    for param in nasa_data.columns:
        if param not in sum(plot_data.values(), []):
            st.subheader(ALL_PARAMETERS[param])
            st.line_chart(nasa_data[param])

    st.subheader("Climate Parameter Correlation Heatmap")
    corr = nasa_data.corr()
    fig_heatmap = go.Figure(data=go.Heatmap(
                   z=corr.values,
                   x=corr.columns,
                   y=corr.columns,
                   colorscale='Viridis'))
    fig_heatmap.update_layout(height=600, width=800, title_text="Parameter Correlation Heatmap")
    st.plotly_chart(fig_heatmap, use_container_width=True)

def analyze_topography(elevation_data):
    st.subheader("Topography Analysis")

    # Calculate slope
    dy, dx = np.gradient(elevation_data)
    slope = np.degrees(np.arctan(np.sqrt(dx*dx + dy*dy)))

    # Calculate aspect
    aspect = np.degrees(np.arctan2(-dx, dy))

    # Create visualizations
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    im1 = ax1.imshow(elevation_data, cmap='terrain')
    ax1.set_title('Elevation')
    plt.colorbar(im1, ax=ax1, label='Elevation (m)')

    im2 = ax2.imshow(slope, cmap='YlOrRd')
    ax2.set_title('Slope')
    plt.colorbar(im2, ax=ax2, label='Slope (degrees)')

    im3 = ax3.imshow(aspect, cmap='hsv')
    ax3.set_title('Aspect')
    plt.colorbar(im3, ax=ax3, label='Aspect (degrees)')

    st.pyplot(fig)

    # Provide insights
    st.subheader("Topography Insights")
    st.write(f"Average Elevation: {np.mean(elevation_data):.2f} meters")
    st.write(f"Average Slope: {np.mean(slope):.2f} degrees")

    # Slope classification
    flat = np.sum((slope >= 0) & (slope < 5)) / slope.size * 100
    gentle = np.sum((slope >= 5) & (slope < 15)) / slope.size * 100
    moderate = np.sum((slope >= 15) & (slope < 30)) / slope.size * 100
    steep = np.sum(slope >= 30) / slope.size * 100

    st.write("Slope Classification:")
    st.write(f"- Flat (0-5°): {flat:.2f}%")
    st.write(f"- Gentle (5-15°): {gentle:.2f}%")
    st.write(f"- Moderate (15-30°): {moderate:.2f}%")
    st.write(f"- Steep (>30°): {steep:.2f}%")

def generate_urban_planning_recommendations(nasa_data, elevation_data):
    st.subheader("Urban Planning Recommendations")

    # Climate-based recommendations
    avg_temp = nasa_data['T2M'].mean() if 'T2M' in nasa_data.columns else None
    avg_precip = nasa_data['PRECTOTCORR'].mean() if 'PRECTOTCORR' in nasa_data.columns else None
    avg_wind = nasa_data['WS2M'].mean() if 'WS2M' in nasa_data.columns else None

    st.write("Climate Considerations:")
    if avg_temp is not None:
        if avg_temp > 25:
            st.write("- Consider heat-resistant building materials and designs")
            st.write("- Plan for ample green spaces and water features for cooling")
        elif avg_temp < 10:
            st.write("- Focus on insulation and heating efficiency in buildings")
            st.write("- Plan for snow removal and ice management in public spaces")

    if avg_precip is not None:
        if avg_precip > 5:
            st.write("- Implement robust drainage systems and flood control measures")
            st.write("- Use permeable pavements to manage stormwater runoff")
        elif avg_precip < 1:
            st.write("- Implement water conservation measures and drought-resistant landscaping")
            st.write("- Consider rainwater harvesting systems for buildings")

    if avg_wind is not None:
        if avg_wind > 5:
            st.write("- Design wind-resistant structures and consider wind barriers")
            st.write("- Explore potential for wind energy generation")
    else:
        st.write("- Wind speed data not available. Consider collecting local wind data for better planning.")

    # Topography-based recommendations
    avg_elevation = np.mean(elevation_data)
    avg_slope = np.mean(np.gradient(elevation_data)[0])

    st.write("Topography Considerations:")
    if avg_slope > 15:
        st.write("- Implement erosion control measures on steeper slopes")
        st.write("- Consider terracing for building sites on slopes")
    else:
        st.write("- The relatively flat terrain is suitable for most urban development")

    if avg_elevation > 1000:
        st.write("- Account for altitude effects on air pressure and temperature")
        st.write("- Ensure infrastructure can handle potential extreme weather at higher elevations")

    st.write("General Recommendations:")
    st.write("- Conduct detailed environmental impact assessments before major developments")
    st.write("- Prioritize sustainable and energy-efficient building designs")
    st.write("- Plan for mixed-use developments to reduce transportation needs")
    st.write("- Incorporate green infrastructure and nature-based solutions in urban design")

def project_climate_change(nasa_data, years):
    projection = pd.DataFrame()
    for column in nasa_data.columns:
        trend = np.polyfit(range(len(nasa_data)), nasa_data[column], 1)
        projection[column] = [trend[1] + trend[0] * i for i in range(len(nasa_data), len(nasa_data) + 365 * years)]
    projection.index = pd.date_range(start=nasa_data.index[-1] + pd.Timedelta(days=1), periods=365 * years, freq='D')
    return projection

def visualize_climate_projection(nasa_data, initial_years):
    st.write("Climate Change Projection")
    years_projection = st.slider("Select years for projection", 10, 50, initial_years)
    
    try:
        if nasa_data.empty:
            st.error("No climate data available for projection. Please ensure you've selected climate parameters and fetched data.")
            return

        climate_projection = project_climate_change(nasa_data, years_projection)
        
        if climate_projection.empty:
            st.error("Climate projection calculation resulted in empty data. Please check the input data.")
            return

        # Create a more informative visualization using Plotly
        fig = go.Figure()
        for column in climate_projection.columns:
            fig.add_trace(go.Scatter(x=climate_projection.index, y=climate_projection[column],
                                     mode='lines', name=ALL_PARAMETERS.get(column, column)))
        
        fig.update_layout(title=f'Climate Projection for {years_projection} Years',
                          xaxis_title='Year',
                          yaxis_title='Value',
                          legend_title='Parameters')
        
        st.plotly_chart(fig)
        
        st.write("Note: This projection is based on linear trends and should be interpreted cautiously.")
        
        # Display summary statistics
        st.write("Projection Summary Statistics:")
        st.write(climate_projection.describe())
        
    except Exception as e:
        st.error(f"An error occurred while projecting climate data: {str(e)}")
        st.write("Please ensure you have selected sufficient historical data for projection.")
        st.write("Debug information:")
        st.write(f"NASA data shape: {nasa_data.shape}")
        st.write(f"NASA data columns: {nasa_data.columns}")
        st.write(f"Years projection: {years_projection}")
        
def analyze_urban_heat_island(nasa_data, elevation_data):
    uhi_effect = np.zeros_like(elevation_data)
    avg_temp = nasa_data['T2M'].mean()
    elevation_range = np.ptp(elevation_data)

    for i in range(elevation_data.shape[0]):
        for j in range(elevation_data.shape[1]):
            uhi_effect[i, j] = avg_temp + (elevation_data[i, j] - elevation_data.min()) / elevation_range * 5

    fig = go.Figure(data=go.Heatmap(z=uhi_effect, colorscale='RdYlBu_r'))
    fig.update_layout(title='Simulated Urban Heat Island Effect', height=600, width=800)
    return fig

def calculate_sustainability_score(nasa_data, elevation_data):
    climate_score = (
        (1 - abs(nasa_data['T2M'].mean() - 20) / 20) * 0.3 +
        (1 - abs(nasa_data['PRECTOTCORR'].mean() - 2) / 2) * 0.2 +
        (nasa_data['ALLSKY_SFC_SW_DWN'].mean() / 300) * 0.2
    )

    topography_score = (
        (1 - np.mean(np.gradient(elevation_data)[0]) / 45) * 0.2 +
        (1 - abs(np.mean(elevation_data) - 500) / 500) * 0.1
    )

    return (climate_score + topography_score) * 10

def export_data(nasa_data, elevation_data):
    nasa_csv = nasa_data.to_csv(index=False).encode('utf-8')
    elevation_csv = pd.DataFrame(elevation_data).to_csv(index=False).encode('utf-8')

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Climate Data CSV",
            data=nasa_csv,
            file_name="climate_data.csv",
            mime="text/csv",
            key="climate_data"
        )
    with col2:
        st.download_button(
            label="Download Elevation Data CSV",
            data=elevation_csv,
            file_name="elevation_data.csv",
            mime="text/csv",
            key="elevation_data"
        )

def create_3d_visualization(elevation_data):
    st.subheader("3D Topography Visualization")
    
    # Create a grid of coordinates
    y, x = np.mgrid[0:elevation_data.shape[0], 0:elevation_data.shape[1]]
    
    # Create the 3D surface plot
    fig = go.Figure(data=[go.Surface(z=elevation_data, x=x, y=y)])
    
    # Update the layout for better visibility
    fig.update_layout(
        title='3D Topography Visualization',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Elevation',
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        width=800,
        height=600
    )
    
    # Display the 3D visualization
    st.plotly_chart(fig)

def main():
    st.title("Urban Planning Map Tool - Pakistan")

    st.sidebar.header("Settings")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start date", datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End date", datetime.now())

    if start_date > end_date:
        st.sidebar.error("Error: End date must fall after start date.")
        return

    st.sidebar.subheader("Climate Parameters")
    selected_params = st.sidebar.multiselect(
        "Select climate parameters",
        list(ALL_PARAMETERS.keys()),
        default=list(ALL_PARAMETERS.keys())[:7],
        format_func=lambda x: ALL_PARAMETERS[x]
    )


    with st.expander("Instructions", expanded=False):
        st.write("1. Use the sidebar to select date range and climate parameters.")
        st.write("2. Draw a polygon or rectangle on the map to select your area of interest.")
        st.write("3. Click 'Analyze Selected Area' to fetch and analyze data.")
        st.write("4. Explore the generated visualizations and recommendations.")
    
    if not selected_params:
        st.warning("Please select at least one climate parameter.")
        return

    st.subheader("Select Area of Interest")

    default_location = [30.3753, 69.3451]  # Center of Pakistan
    m = folium.Map(location=default_location, zoom_start=12)

    draw = Draw(
        draw_options={
            'polyline': False,
            'polygon': True,
            'rectangle': True,
            'circle': False,
            'marker': False,
            'circlemarker': False,
        },
        edit_options={'edit': False}
    )
    draw.add_to(m)
    

    map_data = st_folium(m, width="100%", height=400)


    if map_data and 'all_drawings' in map_data and map_data['all_drawings'] and len(map_data['all_drawings']) > 0:
        geometry = map_data['all_drawings'][-1]['geometry']
        if geometry['type'] in ['Polygon', 'Rectangle']:
            st.session_state.geometry = geometry
            st.success("Area selected. You can now analyze the data.")
            
            if st.button("Analyze Selected Area"):
                coordinates = st.session_state.geometry['coordinates']
                lat, lon = calculate_centroid(coordinates)
                south, west = min(coord[1] for coord in coordinates[0]), min(coord[0] for coord in coordinates[0])
                north, east = max(coord[1] for coord in coordinates[0]), max(coord[0] for coord in coordinates[0])

                progress_bar = st.progress(0)
                
                with st.spinner("Fetching and analyzing data..."):
                    nasa_data = get_nasa_power_data(lat, lon, start_date, end_date, selected_params)
                    progress_bar.progress(50)
                    
                    elevation_data = get_opentopography_data(south, north, west, east)
                    progress_bar.progress(100)

                if nasa_data is not None and elevation_data is not None:
                    st.success("Data fetched and analyzed successfully!")
                    
                    tab1, tab2, tab3= st.tabs(["Climate Analysis", "Topography Analysis", "Urban Planning Recommendations"])
                    
                    with tab1:
                        create_climate_visualizations(nasa_data)
                    
                    with tab2:
                        create_3d_visualization(elevation_data)
                        
                        st.write("Area Sustainability Score")
                        sustainability_score = calculate_sustainability_score(nasa_data, elevation_data)
                        st.metric("Sustainability Score", f"{sustainability_score:.2f}/10")
                        
                        st.subheader("Export Data")
                        export_data(nasa_data, elevation_data)
                        
                        analyze_topography(elevation_data)

                    
                    with tab3:
                        generate_urban_planning_recommendations(nasa_data, elevation_data)

                else:
                    st.error("Failed to fetch or process data. Please try again.")
            
    else:
        st.info("Please draw a polygon or rectangle on the map to select an area.")

if __name__ == "__main__":
    main()