
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from typing import Dict, List
from pyproj import Transformer

# -----------------------------------------------------------------------------
# Page & Auth
# -----------------------------------------------------------------------------
st.set_page_config(page_title="UK GP → LSOA Classification Viewer", layout="wide")

# Raw passwords stored in Streamlit Secrets, e.g.:
# [users]
# duran = "duran"
# carl = "carl"
# bobby = "bobby"
USERS = st.secrets.get("users", {})  # dict: {username: password}

def verify(username: str, password: str) -> bool:
    """Simple raw password comparison (development setup)."""
    return username in USERS and USERS[username] == password

def login_gate():
    """Render a login screen and stop execution until authenticated."""
    if st.session_state.get("logged_in"):
        return  # Already authenticated; proceed to the app

    st.title("Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Sign in"):
        if verify(u, p):
            st.session_state["logged_in"] = True
            st.session_state["user"] = u
            st.success("Authenticated. Loading app...")
            # Use the stable API (replaces deprecated st.experimental_rerun)
            st.rerun()
        else:
            st.error("Invalid credentials")

    # Halt the rest of the script until user logs in
    st.stop()

# Gate the app immediately
login_gate()

# Optional logout button
with st.sidebar:
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()

# -----------------------------------------------------------------------------
# Data loaders (cached)
# -----------------------------------------------------------------------------
DATA_DIR = "data"

@st.cache_data(show_spinner=False)
def load_gp():
    # GP.csv: GP Code, GP Name, Patients, Ward code, Ward Name, LA Code, LA Name
    df = pd.read_csv(f"{DATA_DIR}/GP.csv")
    df.columns = [c.strip() for c in df.columns]
    df["GP Code"] = df["GP Code"].astype(str).str.strip().str.upper()
    df["GP Name"] = df["GP Name"].astype(str).str.strip()
    return df

@st.cache_data(show_spinner=False)
def load_gp_lsoa():
    # GP_LSOA.csv: Rank link, GP, LSOA, Patients, % total, Rank, M, F
    df = pd.read_csv(f"{DATA_DIR}/GP_LSOA.csv")
    df.columns = [c.strip() for c in df.columns]
    # Normalize keys and numeric fields
    if "GP" in df.columns:
        df["GP"] = df["GP"].astype(str).str.strip().str.upper()
    if "LSOA" in df.columns:
        df["LSOA"] = df["LSOA"].astype(str).str.strip().str.upper()
    if "Patients" in df.columns:
        df["Patients"] = pd.to_numeric(df["Patients"], errors="coerce").fillna(0)
    return df

@st.cache_data(show_spinner=False)
def load_classification():
    # LSOA_Classification.csv: includes LSOA and many Pop*/HH* columns
    df = pd.read_csv(f"{DATA_DIR}/LSOA_Classification.csv")
    df.columns = [c.strip() for c in df.columns]
    if "LSOA" not in df.columns:
        raise ValueError("LSOA_Classification.csv must have a 'LSOA' column")
    df["LSOA"] = df["LSOA"].astype(str).str.strip().str.upper()
    return df

@st.cache_data(show_spinner=False)
def load_centroids():
    """
    LSOA_PopCentroids_EW_2021_V4.csv: columns include LSOA21CD, x, y
    - If x,y look like lon/lat, keep as-is
    - Otherwise assume British National Grid (EPSG:27700) and convert to WGS84 (EPSG:4326)
    """
    df = pd.read_csv(f"{DATA_DIR}/LSOA_PopCentroids_EW_2021_V4.csv")
    df.columns = [c.strip() for c in df.columns]
    for required in ["LSOA21CD", "x", "y"]:
        if required not in df.columns:
            raise ValueError("Centroid CSV must have columns: LSOA21CD, x, y")
    df["LSOA21CD"] = df["LSOA21CD"].astype(str).str.strip().str.upper()
    x = pd.to_numeric(df["x"], errors="coerce")
    y = pd.to_numeric(df["y"], errors="coerce")

    # Heuristic to detect lon/lat vs BNG
    looks_like_lonlat = (
        x.between(-10, 10).mean() > 0.90 and
        y.between(49, 60).mean() > 0.90
    )

    if looks_like_lonlat:
        df["lon"] = x
        df["lat"] = y
    else:
        transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
        lons, lats = transformer.transform(x.values, y.values)
        df["lon"] = lons
        df["lat"] = lats

    return df[["LSOA21CD", "lat", "lon"]].rename(columns={"LSOA21CD": "LSOA"})

gp = load_gp()
gp_lsoa = load_gp_lsoa()
lsoa_cls = load_classification()
centroids = load_centroids()

# -----------------------------------------------------------------------------
# Classification grouping (Major / Sub / Micro)
# -----------------------------------------------------------------------------
def find_columns_by_prefix(df: pd.DataFrame, prefix: str) -> List[str]:
    return [c for c in df.columns if c.startswith(prefix)]

def classification_groups(df: pd.DataFrame, base: str, granularity: str) -> Dict[str, List[str]]:
    """
    Build groups from classification columns for base ('Pop' or 'HH'):
      - Major: base + digits (Pop1..Pop8)
      - Sub:   base + digit + letter (Pop1a..Pop8c), summing any micro columns under it
      - Micro: base + digit + letter + digits (Pop1a1..)
    Returns {group_name: [columns to sum]}
    """
    cols = find_columns_by_prefix(df, base)
    tokens = [c[len(base):] for c in cols]  # suffix after 'Pop' or 'HH'

    majors = sorted({t for t in tokens if t.isdigit()})
    subs   = sorted({t for t in tokens if len(t) >= 2 and t[0].isdigit() and t[1].isalpha() and not t[1:].isdigit()})
    micros = sorted({t for t in tokens if len(t) >= 3 and t[0].isdigit() and t[1].isalpha() and t[2:].isdigit()})

    groups: Dict[str, List[str]] = {}

    if granularity == "Major" and majors:
        for m in majors:
            col = f"{base}{m}"
            if col in cols:
                groups[col] = [col]
    elif granularity == "Sub" and subs:
        for s in subs:
            cands = [f"{base}{s}"] if f"{base}{s}" in cols else []
            cands += [f"{base}{t}" for t in micros if t.startswith(s)]
            cands = [c for c in cands if c in cols]
            if cands:
                groups[f"{base}{s}"] = cands
    else:  # Micro
        if micros:
            for mi in micros:
                col = f"{base}{mi}"
                if col in cols:
                    groups[col] = [col]
        elif majors:  # Fallback to major
            for m in majors:
                col = f"{base}{m}"
                if col in cols:
                    groups[col] = [col]

    return groups

# -----------------------------------------------------------------------------
# Sidebar controls (defaults: Population + Major)
# -----------------------------------------------------------------------------
st.sidebar.header("Controls")

weighting = st.sidebar.radio("Weighting method", ["Population (Pop*)", "Households (HH*)"], index=0)
base = "Pop" if "Population" in weighting else "HH"

granularity = st.sidebar.radio("Classification granularity", ["Major", "Sub", "Micro"], index=0)

groups = classification_groups(lsoa_cls, base=base, granularity=granularity)
if not groups:
    st.error(f"No classification groups found for base '{base}' and granularity '{granularity}'. Check LSOA_Classification columns.")
    st.stop()

# GP search
st.sidebar.write("---")
st.sidebar.caption("Search GP by name")
query = st.sidebar.text_input("Type GP name (or code, e.g., 'A81001')")
matches = gp[gp["GP Name"].str.contains(query, case=False, na=False)].sort_values("GP Name")

if matches.empty and query:
    st.sidebar.warning("No GP name matches. Try another term.")

selected_gp_name = st.sidebar.selectbox("Select a GP", options=(matches["GP Name"] if not matches.empty else gp["GP Name"]))
selected_row = gp[gp["GP Name"] == selected_gp_name].iloc[0]
selected_gp_code = selected_row["GP Code"]

# -----------------------------------------------------------------------------
# Filter GP_LSOA using GP code (your sample shows GP is like 'A81001')
# -----------------------------------------------------------------------------
gpl = gp_lsoa[gp_lsoa["GP"] == selected_gp_code].copy()

if gpl.empty:
    # Fallback: try matching by name substring
    gpl = gp_lsoa[gp_lsoa["GP"].str.contains(selected_gp_name, case=False, na=False)].copy()

if gpl.empty:
    st.error("No LSOA rows found for this GP in GP_LSOA.csv. Check if 'GP' holds the GP Code or Name.")
    st.stop()

# -----------------------------------------------------------------------------
# Build classification counts by LSOA & estimate patients per classification
# -----------------------------------------------------------------------------
# Sum selected group columns for each LSOA
records = []
for grp_name, cols in groups.items():
    # ensure numeric
    for c in cols:
        lsoa_cls[c] = pd.to_numeric(lsoa_cls[c], errors="coerce").fillna(0)
    tmp = lsoa_cls[["LSOA"]].copy()
    tmp["count"] = lsoa_cls[cols].sum(axis=1)
    tmp["classification"] = grp_name
    records.append(tmp)

cls_tidy = pd.concat(records, ignore_index=True)

# Denominator by LSOA
denom = cls_tidy.groupby("LSOA")["count"].sum().rename("denom").reset_index()

# Merge proportions
dist = cls_tidy.merge(denom, on="LSOA", how="left")
dist = dist[dist["denom"] > 0].copy()
dist["prop"] = dist["count"] / dist["denom"]

# Merge GP patients by LSOA
gpl_pat = gpl[["LSOA", "Patients"]].copy()
dist = dist.merge(gpl_pat, on="LSOA", how="inner")

# Estimated patients per classification
dist["est_patients"] = dist["Patients"] * dist["prop"]

# Top 5 classifications
top_df = dist.groupby("classification")["est_patients"].sum().sort_values(ascending=False).reset_index()
top5 = top_df.head(5)
top5_names = set(top5["classification"])

# Per-LSOA contributions for Top 5
per_lsoa_top5 = (
    dist[dist["classification"].isin(top5_names)]
    .groupby(["classification", "LSOA"])["est_patients"]
    .sum()
    .reset_index()
)

# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------
st.title("GP → LSOA Classification Viewer (UK)")
st.caption(f"Signed in as **{st.session_state.get('user', 'user')}**")
st.write(f"**Selected GP:** {selected_gp_name}  |  **GP Code:** {selected_gp_code}")

c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("Top 5 classifications (estimated patients)")
    st.dataframe(
        top5.rename(columns={"classification": "Classification", "est_patients": "Estimated patients"}),
        use_container_width=True
    )
    st.bar_chart(top5.set_index("classification")["est_patients"])

with c2:
    st.subheader("Notes")
    st.markdown(
        f"""
- Weighting method: **{weighting}**  
- Granularity: **{granularity}** (columns starting with **{base}**)  
- Estimation: GP's LSOA patient counts apportioned by LSOA’s {('population' if base=='Pop' else 'household')} distribution across selected classifications.  
- This is an estimator (no patient-level classification).
        """
    )

st.subheader("LSOA contributions for Top 5 classifications")
st.dataframe(
    per_lsoa_top5.sort_values(["classification", "est_patients"], ascending=[True, False])
      .rename(columns={"classification": "Classification", "est_patients": "Estimated patients"}),
    use_container_width=True
)

# -----------------------------------------------------------------------------
# Map (pydeck + centroids)
# -----------------------------------------------------------------------------
st.subheader("Map: LSOAs contributing to Top 5 classifications")

# Total by LSOA for Top 5
lsoa_tot = per_lsoa_top5.groupby("LSOA")["est_patients"].sum().reset_index()

# Join centroids
map_df = lsoa_tot.merge(centroids, on="LSOA", how="left").dropna(subset=["lat", "lon"])
if map_df.empty:
    st.warning("No centroid matches. Ensure LSOA codes align between files.")
else:
    # Scale radius (meters)
    max_val = map_df["est_patients"].max()
    # Avoid division by zero
    scale = 5000 if max_val > 0 else 500
    map_df["radius"] = (map_df["est_patients"] / max_val) * scale + 500 if max_val > 0 else 500

    tooltip_html = """
    <b>LSOA:</b> {LSOA} <br/>
    <b>Est. patients (Top 5 total):</b> {est_patients:.0f}
    """

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position='[lon, lat]',
        get_radius='radius',
        get_fill_color='[200, 30, 0, 160]',
        pickable=True,
    )

    view = pdk.ViewState(latitude=54, longitude=-2, zoom=5)  # UK view
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            tooltip={"html": tooltip_html, "style": {"backgroundColor": "steelblue", "color": "white"}}
        )
    )
