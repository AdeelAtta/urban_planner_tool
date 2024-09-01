import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rasterio.io import MemoryFile
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os
from dotenv import load_dotenv
import openai
from openai import OpenAI

load_dotenv()

# Set page configuration
st.set_page_config(layout="wide", page_title="SmartTown: Urban Planning Tool - Pakistan")

# Securely store API keys
OPENTOPOGRAPHY_API_KEY = os.getenv("OPENTOPOGRAPHY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

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

       # Calculate insights
    avg_elevation = np.mean(elevation_data)
    avg_slope = np.mean(slope)

    # Provide insights
    st.subheader("Topography Insights")
    st.write(f"Average Elevation: {np.mean(elevation_data):.2f} meters")
    st.write(f"Average Slope: {np.mean(slope):.2f} degrees")

    # Slope classification
    flat = np.sum((slope >= 0) & (slope < 5)) / slope.size * 100
    gentle = np.sum((slope >= 5) & (slope < 15)) / slope.size * 100
    moderate = np.sum((slope >= 15) & (slope < 30)) / slope.size * 100
    steep = np.sum(slope >= 30) / slope.size * 100

        # Save data to session state
    st.session_state['topography_analysis'] = {
        'average_elevation': avg_elevation,
        'average_slope': avg_slope,
        'slope_classification': {
            'flat': flat,
            'gentle': gentle,
            'moderate': moderate,
            'steep': steep
        }
    }

    st.write("Slope Classification:")
    st.write(f"- Flat (0-5°): {flat:.2f}%")
    st.write(f"- Gentle (5-15°): {gentle:.2f}%")
    st.write(f"- Moderate (15-30°): {moderate:.2f}%")
    st.write(f"- Steep (>30°): {steep:.2f}%")

def create_3d_visualization(elevation_data):
    st.subheader("3D Topography Visualization")
    
    y, x = np.mgrid[0:elevation_data.shape[0], 0:elevation_data.shape[1]]
    
    fig = go.Figure(data=[go.Surface(z=elevation_data, x=x, y=y)])
    
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
    
    st.plotly_chart(fig)

def evaluate_land_suitability(nasa_data, elevation_data, weights=None, ideal_values=None):
    if not isinstance(nasa_data, pd.DataFrame) or not isinstance(elevation_data, np.ndarray):
        raise TypeError("nasa_data must be a pandas DataFrame and elevation_data must be a numpy array")
    
    required_columns = ['T2M', 'PRECTOTCORR', 'ALLSKY_SFC_SW_DWN']
    if not all(col in nasa_data.columns for col in required_columns):
        raise ValueError(f"nasa_data must contain columns: {', '.join(required_columns)}")
    
    if weights is None:
        weights = {'temp': 0.3, 'precip': 0.35, 'solar': 0.2, 'slope': 0.15}
    
    if ideal_values is None:
        ideal_values = {'temp': 30, 'precip': 1.5, 'solar': 300, 'slope': 30}
    
    temp_suitability = 1 - np.abs(nasa_data['T2M'].mean() - ideal_values['temp']) / 30
    
    precip_suitability = 1 - np.abs(nasa_data['PRECTOTCORR'].mean() - ideal_values['precip']) / 1.5
    
    solar_suitability = np.minimum(nasa_data['ALLSKY_SFC_SW_DWN'].mean() / ideal_values['solar'], 1)
    
    dy, dx = np.gradient(elevation_data)
    slope = np.degrees(np.arctan(np.sqrt(dx*dx + dy*dy)))
    slope_suitability = 1 / (1 + np.exp((slope - ideal_values['slope']) / 10))
    
    suitability_score = (
        temp_suitability * weights['temp'] +
        precip_suitability * weights['precip'] +
        solar_suitability * weights['solar'] +
        slope_suitability * weights['slope']
    )
    
    return suitability_score

def urban_planning_chatbot(nasa_data, elevation_data, user_input, coordinates,topography_analysis):
    # Prepare a summary of the data for the chatbot
    avg_temp = nasa_data['T2M'].mean() if 'T2M' in nasa_data.columns else "N/A"
    avg_precip = nasa_data['PRECTOTCORR'].mean() if 'PRECTOTCORR' in nasa_data.columns else "N/A"
    avg_solar = nasa_data['ALLSKY_SFC_SW_DWN'].mean() if 'ALLSKY_SFC_SW_DWN' in nasa_data.columns else "N/A"
    avg_elevation = np.mean(elevation_data)
    
    # Calculate centroid of the selected area
    lat, lon = calculate_centroid(coordinates)

    # Prepare the conversation context
    conversation = [
        {"role": "system", 
         "content": """
        I am an advanced AI urban planning assistant with expertise in sustainable development, climate-responsive design, and data-driven decision making. I have access to detailed climate and topographical data for a specific area. My role is to provide insightful, practical, and tailored urban planning recommendations based on this data and best practices in urban design.
        Available Data:

        Coordinates: {coordinates}
        Lat {lat:.4f}, Lon {lon:.4f}
        Climate Data:

        Average Temperature: {avg_temp:.2f}°C
        Average Precipitation: {avg_precip:.2f} mm/day
        Average Solar Radiation: {avg_solar:.2f} W/m^2
        {additional_climate_params}


        Topographical Data:

        Average Elevation: {avg_elevation:.2f} meters
        Slope characteristics: {slope_chars}
        Topography Insights: {topography_analysis}

        Land Suitability Score: {land_suitability:.2f}/1.00

        My Capabilities:

        Analyze the provided data to assess the area's potential for various urban development projects.
        Suggest sustainable urban planning strategies that align with the local climate and topography.
        Provide recommendations on:

        Optimal building designs and orientations.
        facing of buildings for ventilations
        Energy-efficient infrastructure
        Water management systems
        Green spaces and urban forestry
        Transportation network planning
        Climate change adaptation measures


        Identify potential challenges or limitations based on the data and suggest mitigation strategies.
        Offer insights on how to balance development needs with environmental conservation.

        My Response Guidelines:

        I structure my responses clearly, using headings or bullet points where appropriate.
        I provide specific, actionable recommendations based on the data provided.
        I explain the reasoning behind my suggestions, referencing the climate and topographical data.
        When relevant, I mention any assumptions I'm making or additional data that would be helpful.
        If asked about topics outside my expertise or data scope, I clearly state the limitations of my knowledge.
        I encourage sustainable and resilient urban planning practices in all my recommendations.

        My goal is to assist urban planners in making informed, data-driven decisions that promote sustainable and livable urban environments. I tailor my responses to the specific context provided in each query also gave specified Answer.
        """
         },
        {"role": "user", "content": f"Here's a summary of the area data:\n Lat {lat:.4f}, Lon {lon:.4f}\nAverage Temperature: {avg_temp:.2f}°C\nAverage Precipitation: {avg_precip:.2f} mm/day\nAverage Solar Radiation: {avg_solar:.2f} W/m^2\nAverage Elevation: {avg_elevation:.2f} meters\n\nBased on this data and location, {user_input}"}
    ]
    
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conversation
        )
        ai_message = response.choices[0].message.content
        
        return ai_message
    except Exception as e:
        return f"An error occurred: {str(e)}"
    

# def display_chat_history():
#     st.markdown("### Chat History")
#     if 'chat_history' in st.session_state and st.session_state.chat_history:
#         for message in st.session_state.chat_history:
#             if message['is_user']:
#                 st.markdown(f"**You:** {message['content']}")
#             else:
#                 st.markdown(f"**AI:** {message['content']}")
#     else:
#         st.markdown("No conversation history yet.")

def create_chat_interface():
    # Chat input form
    user_input = st.text_input("Enter your urban planning query...", key="user_input")
    submit_button = st.button(label="Submit Query")

    if submit_button and user_input:
        send_message(user_input)

    # Custom CSS for modern and minimalistic chat bubbles
    st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .chat-bubble {
        padding: 12px 16px;
        border-radius: 18px;
        max-width: 80%;
        line-height: 1.4;
        font-size: 16px;
    }
    .user-bubble {
        align-self: flex-start;
        background-color: #f0f0f0;
        color: #333;
    }
    .ai-bubble {
        align-self: flex-start;
        background-color: #e1f5fe;
        color: #0277bd;
    }
    .message-container {
        display: flex;
        flex-direction: column;
    }
    .timestamp {
        font-size: 12px;
        color: #888;
        margin-top: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Display chat history
    st.markdown("### Chat History")
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            bubble_class = "user-bubble" if message['is_user'] else "ai-bubble"
            sender = "You" if message['is_user'] else "AI"
            st.markdown(f"""
            <div class="message-container" style="align-items: {'flex-end' if message['is_user'] else 'flex-start'}">
                <div class="chat-bubble {bubble_class}">{message['content']}</div>
                <div class="timestamp">{sender} • Just now</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("No conversation history yet.")


def send_message(user_input):
    with st.spinner("AI is thinking..."):
        ai_response = urban_planning_chatbot(st.session_state.nasa_data, st.session_state.elevation_data, user_input, st.session_state.coordinates,st.session_state['topography_analysis'])
    st.session_state.chat_history.insert(0,{"is_user": False, "content": ai_response})
    st.session_state.chat_history.insert(0,{"is_user": True, "content": user_input})

def main():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    st.title("SmartTown: Optimal Land Selection for Urban Planning")

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

    if not selected_params:
        st.warning("Please select at least one climate parameter.")
        return

    st.subheader("Select Area of Interest")

    default_location = [33.6844, 73.0479]  # Coordinates for Islamabad
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
            
            if st.button("Analyze Selected Area",key="selectedArea"):
                coordinates = st.session_state.geometry['coordinates']
                lat, lon = calculate_centroid(coordinates)
                south, west = min(coord[1] for coord in coordinates[0]), min(coord[0] for coord in coordinates[0])
                north, east = max(coord[1] for coord in coordinates[0]), max(coord[0] for coord in coordinates[0])

                progress_bar = st.progress(0)
                
                with st.spinner("Fetching and analyzing data..."):
                    with ThreadPoolExecutor() as executor:
                        nasa_future = executor.submit(get_nasa_power_data, lat, lon, start_date, end_date, selected_params)
                        elevation_future = executor.submit(get_opentopography_data, south, north, west, east)

                        nasa_data = nasa_future.result()
                        elevation_data = elevation_future.result()
                    
                    st.session_state.nasa_data = nasa_data
                    st.session_state.elevation_data = elevation_data
                    st.session_state.coordinates = coordinates
                    
                    progress_bar.progress(100)

                if nasa_data is not None and elevation_data is not None:
                    st.success("Data fetched and analyzed successfully!")
                else:
                    st.error("Failed to fetch or process data. Please try again.")

    if 'nasa_data' in st.session_state and 'elevation_data' in st.session_state:
        tab1, tab2, tab3, tab4 = st.tabs(["Climate Analysis", "Topography Analysis", "Land Suitability", "Urban Planning Recommendations"])
        
        with tab1:
            create_climate_visualizations(st.session_state.nasa_data)

        with tab2:
            create_3d_visualization(st.session_state.elevation_data)
            analyze_topography(st.session_state.elevation_data)

        with tab3:
            st.subheader("Land Suitability Analysis")
            
            # Allow user to customize weights and ideal values
            st.subheader("Customize Land Suitability Parameters")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Weights (must sum to 1)")
                weight_temp = st.slider("Temperature Weight", 0.0, 1.0, 0.3, 0.05)
                weight_precip = st.slider("Precipitation Weight", 0.0, 1.0, 0.35, 0.05)
                weight_solar = st.slider("Solar Radiation Weight", 0.0, 1.0, 0.2, 0.05)
                weight_slope = st.slider("Slope Weight", 0.0, 1.0, 0.15, 0.05)
            with col2:
                st.write("Ideal Values")
                ideal_temp = st.slider("Ideal Temperature (°C)", 0, 40, 30)
                ideal_precip = st.slider("Ideal Precipitation (mm/day)", 0.0, 10.0, 1.5, 0.1)
                ideal_solar = st.slider("Ideal Solar Radiation (W/m^2)", 0, 500, 300, 10)
                ideal_slope = st.slider("Ideal Slope (%)", 0, 45, 30)

            weights = {
                'temp': weight_temp,
                'precip': weight_precip,
                'solar': weight_solar,
                'slope': weight_slope
            }
            ideal_values = {
                'temp': ideal_temp,
                'precip': ideal_precip,
                'solar': ideal_solar,
                'slope': ideal_slope
            }

            suitability_score = evaluate_land_suitability(st.session_state.nasa_data, st.session_state.elevation_data, weights, ideal_values)
            fig = go.Figure(data=go.Heatmap(z=suitability_score, colorscale='RdYlGn'))
            fig.update_layout(title='Land Suitability Heatmap', height=600, width=800)
            st.plotly_chart(fig)
                
            st.write("Suitability Score Legend:")
            st.write("- Green: More suitable for urban development")
            st.write("- Yellow: Moderately suitable")
            st.write("- Red: Less suitable, may require additional considerations")
                
            avg_suitability = np.mean(suitability_score)
            st.metric("Average Land Suitability Score", f"{avg_suitability:.2f}/1.00")

        with tab4:
            st.header("Urban Planning Assistant")
            st.markdown("Discuss your urban planning queries and receive expert advice.")
            # display_chat_history()
            create_chat_interface()
                                                
    else:
        st.info("Please draw a polygon or rectangle on the map and click 'Analyze Selected Area' to see the results.")

if __name__ == "__main__":
    main()