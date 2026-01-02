
import os
import tempfile
import streamlit as st
import geopandas as gpd

# If you prefer gdown (handles large public Drive files & confirmation pages), uncomment:
# import gdown  # and add 'gdown' to requirements.txt

st.set_page_config(page_title="Geographical Area Summary Tool", layout="centered")
st.title("Geographical Area Summary Tool")

# ---- Configuration: choose ONE data source via secrets or local fallback ----
DRIVE_FILE_ID = st.secrets.get("DRIVE_FILE_ID")         # e.g., "1abc...xyz"
DIRECT_URL    = st.secrets.get("DRIVE_DIRECT_URL")      # e.g., https://drive.google.com/uc?export=download&id=...
LOCAL_FILE    = "LSOA_IMD2025_WGS84.gpkg"               # for local development

@st.cache_data(ttl=900)
def _download_gpkg_to_tmp(file_id: str = None, url: str = None) -> str:
    """
    Download a GeoPackage to a temporary file and return the path.
    If a previously downloaded file exists, reuse it.
    """
    tmpdir = tempfile.gettempdir()
    out_path = os.path.join(tmpdir, "LSOA_IMD2025_WGS84.gpkg")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    # Option A: Google Drive by File ID using direct download endpoint
    if file_id:
        dl_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        # Robust method with gdown (uncomment if you add gdown):
        # gdown.download(id=file_id, output=out_path, quiet=True)
        # return out_path

        # Simple method using requests (ok for many files; for very large files prefer gdown)
        import requests
        r = requests.get(dl_url, stream=True, timeout=60)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        return out_path

    # Option B: a direct HTTPS URL
    if url:
        import requests
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        return out_path

    # Option C: local file fallback (development)
    if os.path.exists(LOCAL_FILE):
        return os.path.abspath(LOCAL_FILE)

    raise FileNotFoundError(
        "No data source configured. Provide DRIVE_FILE_ID or DRIVE_DIRECT_URL via st.secrets, "
        "or place the GeoPackage locally."
    )

@st.cache_data(ttl=900)
def load_data() -> gpd.GeoDataFrame:
    """
    Download (if needed) and read the GeoPackage into EPSG:3857 (meters) for distance ops.
    """
    gpkg_path = _download_gpkg_to_tmp(file_id=DRIVE_FILE_ID, url=DIRECT_URL)
    # GeoPandas will use pyogrio if available (recommended), otherwise Fiona.
    gdf = gpd.read_file(gpkg_path)
    # Project to meters for distance calculations
    gdf = gdf.to_crs(epsg=3857)
    return gdf

def summarize_nearby_areas(gdf: gpd.GeoDataFrame, selected_name: str, distance_km: int) -> gpd.GeoDataFrame:
    """
    Buffer the selected area by distance (km) and return intersecting LSOAs.
    """
    row = gdf.loc[gdf['LSOA21NM'] == selected_name]
    if row.empty:
        return gdf.iloc[0:0]
    selected_geom = row.geometry.iloc[0]
    buffer = selected_geom.buffer(distance_km * 1000)  # km -> meters
    nearby = gdf[gdf.intersects(buffer)]
    return nearby

# ---- UI & workflow ----
try:
    gdf = load_data()
    st.write(f"Total areas loaded: {len(gdf)}")

    # Optional: show full map (centroids)
    if st.checkbox("Show map of all areas"):
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        # Centroids (note: centroid in WGS84 is fine for display)
        gdf_wgs84["lon"] = gdf_wgs84.geometry.centroid.x
        gdf_wgs84["lat"] = gdf_wgs84.geometry.centroid.y
        st.map(gdf_wgs84[["lat", "lon"]])

    # Controls
    selected_area = st.selectbox("Select an Area", sorted(gdf['LSOA21NM'].unique()))
    # Fix slider default to be within the min/max
    distance = st.slider("Distance (km)", min_value=20, max_value=30, value=25, step=1)

    # Action
    if st.button("Summarize Nearby Areas"):
        result = summarize_nearby_areas(gdf, selected_area, distance)
        st.write(f"Found {len(result)} areas within {distance} km of {selected_area}")
        st.dataframe(result[['LSOA21NM']])

        # Optional: show result map
        if not result.empty and st.checkbox("Show map of nearby areas"):
            result_wgs84 = result.to_crs(epsg=4326)
            result_wgs84["lon"] = result_wgs84.geometry.centroid.x
            result_wgs84["lat"] = result_wgs84.geometry.centroid.y
            st.map(result_wgs84[["lat", "lon"]])

except Exception as e:
    st.error(f"Error loading GeoPackage file: {e}")
