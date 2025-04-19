import streamlit as st
import requests
import io
import os
from PIL import Image
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
from opencage.geocoder import OpenCageGeocode
import overpy
from math import radians, cos, sin, asin, sqrt
from dotenv import load_dotenv
load_dotenv()


# Get API key from environment
OPENCAGE_API_KEY = os.getenv("OPEN_CAGE_API")
geocoder = OpenCageGeocode(OPENCAGE_API_KEY)
api = overpy.Overpass()

# Haversine distance function
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    return R * c

# Recycling center fetcher
def get_recycling_centers(latitude, longitude, radius=5000):
    query = f"""
    [out:json];
    node["amenity"="recycling"](around:{radius},{latitude},{longitude});
    out body;
    """
    try:
        result = api.query(query)
        centers = []
        for node in result.nodes:
            center_lat = float(node.lat)
            center_lon = float(node.lon)
            distance_km = haversine(float(latitude), float(longitude), center_lat, center_lon)
            name = node.tags.get("name", "Unnamed Recycling Center")
            phone = node.tags.get("phone") or node.tags.get("contact:phone") or node.tags.get("telephone") or "Not available"

            centers.append({
                "name": name,
                "lat": center_lat,
                "lon": center_lon,
                "phone": phone,
                "distance_km": round(distance_km, 2)
            })

        return sorted(centers, key=lambda x: x["distance_km"])
    except Exception as e:
        st.error(f"Error fetching recycling centers: {e}")
        return []

# Streamlit UI setup
st.set_page_config(page_title="E-Waste Analyzer", page_icon="♻️", layout="centered")

st.markdown("""
    <style>
        .st-success {background-color: #2d6a4f !important; color: white !important; border-radius: 10px; padding: 10px;}
        .st-info {background-color: #1b4965 !important; color: white !important; border-radius: 10px; padding: 10px;}
        .st-warning {background-color: #ffb703 !important; color: black !important; border-radius: 10px; padding: 10px;}
        .st-error {background-color: #d90429 !important; color: white !important; border-radius: 10px; padding: 10px;}
    </style>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🧪 Analyze E-Waste", "📍 Find Recycling Center"])

# Tab 1: E-Waste Analyzer
with tab1:
    st.title("♻️ E-Waste Analyzer")
    uploaded_file = st.file_uploader("Upload an E-Waste Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        try:
            image_obj = Image.open(uploaded_file)
            st.image(image_obj, caption="Uploaded Image", use_container_width=True)

            img_bytes = io.BytesIO()
            image_obj.save(img_bytes, format="PNG")
            img_bytes = img_bytes.getvalue()

            with st.spinner("Analyzing E-Waste... 🔄"):
                response = requests.post("http://127.0.0.1:8000/predict/", files={"file": ("image.png", img_bytes, "image/png")})

            if response.status_code == 200:
                result = response.json()
                predicted_class = result.get("predicted_class", "Unknown")
                confidence_score = float(result.get("confidence", "0").strip('%'))
                confidence = f"{confidence_score:.2f}%"
                recyclability = result.get("recyclability", "Unknown")

                st.markdown(f"<div class='st-success'><b>Prediction:</b> {predicted_class}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='st-info'><b>Confidence:</b> {confidence}</div>", unsafe_allow_html=True)

                if recyclability.lower() == "good":
                    st.markdown(f"<div class='st-success'><b>Recyclability:</b> Good ♻️</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='st-warning'><b>Recyclability:</b> {recyclability}%</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='st-error'>❌ Error: Could not get prediction from server.</div>", unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f"<div class='st-error'>⚠️ Exception: {str(e)}</div>", unsafe_allow_html=True)

# Tab 2: Recycling Center Finder
with tab2:
    st.header("📍 Find the Nearest E-Waste Recycling Center (OpenStreetMap)")
    user_location = st.text_input("Enter your location (City, ZIP, or Address):")

    if user_location:
        try:
            results = geocoder.geocode(user_location)
            if results and len(results):
                geometry = results[0]['geometry']
                latitude = float(geometry['lat'])
                longitude = float(geometry['lng'])
                formatted_address = results[0]['formatted']

                st.success(f"Location found: {formatted_address}")

                recycling_centers = get_recycling_centers(latitude, longitude)
                if not recycling_centers:
                    st.warning("No recycling centers found nearby.")

                map_center = folium.Map(location=[latitude, longitude], zoom_start=13)
                folium.Marker([latitude, longitude], tooltip="Your Location", icon=folium.Icon(color="blue")).add_to(map_center)

                for center in recycling_centers:
                    folium.Marker(
                        [center["lat"], center["lon"]],
                        tooltip=center["name"],
                        popup=folium.Popup(
                            f"""
                            <b>{center['name']}</b><br>
                            📞 Phone: {center['phone']}<br>
                            📍 Distance: {center['distance_km']} km
                            """,
                            max_width=300
                        ),
                        icon=folium.Icon(color="green", icon="recycle", prefix="fa")
                    ).add_to(map_center)

                st_folium(map_center, width=700, height=500)
            else:
                st.error("Location not found. Try another input.")
        except Exception as e:
            st.error(f"Error: {e}")
