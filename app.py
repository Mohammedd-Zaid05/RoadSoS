import math
import os
import pickle
import re
import tempfile
from pathlib import Path

import folium
import google.generativeai as genai
import numpy as np
import osmnx as ox
import pandas as pd
import spacy
import streamlit as st
import streamlit.components.v1 as components
import whisper
from dotenv import load_dotenv
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from spacy.matcher import PhraseMatcher

from thefuzz import fuzz

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

try:
    from streamlit_mic_recorder import mic_recorder
    MIC_AVAILABLE = True
except Exception:
    mic_recorder = None
    MIC_AVAILABLE = False

try:
    from streamlit_geolocation import streamlit_geolocation
    GEOLOCATION_AVAILABLE = True
except Exception:
    streamlit_geolocation = None
    GEOLOCATION_AVAILABLE = False

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
MODELS_DIR = APP_DIR / "models"

FEATURES = [
    "Road Accidents - Cases",
    "Road Accidents - Injured",
    "Road Accidents - Died",
    "Railway Accidents - Cases",
    "Railway Accidents - Died",
]

CHENNAI_AREAS = [
    "Adyar", "Alandur", "Alapakkam", "Alwarpet", "Alwarthirunagar", "Ambattur",
    "Aminjikarai", "Anna Nagar", "Annanur", "Arumbakkam", "Ashok Nagar", "Avadi",
    "Ayanavaram", "Beemannapettai", "Besant Nagar", "Basin Bridge", "Chepauk",
    "Chetput", "Chintadripet", "Chitlapakkam", "Choolai", "Choolaimedu", "Chrompet",
    "Egmore", "Ekkaduthangal", "Eranavur", "Ennore", "Foreshore Estate",
    "Fort St. George", "George Town", "Gopalapuram", "Government Estate", "Guindy",
    "Guduvancheri", "IIT Madras", "Injambakkam", "ICF", "Iyyapanthangal",
    "Jafferkhanpet", "Kadambathur", "Karapakkam", "Kattivakkam", "Kattupakkam",
    "Kazhipattur", "K.K. Nagar", "Keelkattalai", "Kilpauk", "Kodambakkam",
    "Kodungaiyur", "Kolathur", "Korattur", "Korukkupet", "Kottivakkam",
    "Kotturpuram", "Kottur", "Kovur", "Koyambedu", "Kundrathur", "Madhavaram",
    "Madhavaram Milk Colony", "Madipakkam", "Madambakkam", "Maduravoyal", "Manali",
    "Manali New Town", "Manapakkam", "Mandaveli", "Mangadu", "Mannadi", "Mathur",
    "Medavakkam", "Meenambakkam", "MGR Nagar", "Minjur", "Mogappair", "MKB Nagar",
    "Mount Road", "Moolakadai", "Moulivakkam", "Mugalivakkam", "Mudichur",
    "Mylapore", "Nandanam", "Nanganallur", "Nanmangalam", "Neelankarai",
    "Nemilichery", "Nesapakkam", "Nolambur", "Noombal", "Nungambakkam", "Otteri",
    "Padi", "Pakkam", "Palavakkam", "Pallavaram", "Pallikaranai", "Pammal",
    "Park Town", "Parry's Corner", "Pattabiram", "Pattaravakkam", "Pazhavanthangal",
    "Peerkankaranai", "Perambur", "Peravallur", "Perumbakkam", "Perungalathur",
    "Perungudi", "Pozhichalur", "Poonamallee", "Porur", "Pudupet", "Pulianthope",
    "Purasaiwalkam", "Puthagaram", "Puzhal", "Puzhuthivakkam", "Raj Bhavan",
    "Ramavaram", "Red Hills", "Royapettah", "Royapuram", "Saidapet", "Saligramam",
    "Santhome", "Sembakkam", "Selaiyur", "Sithalapakkam", "Shenoy Nagar",
    "Sholavaram", "Sholinganallur", "Sikkarayapuram", "Sowcarpet",
    "St.Thomas Mount", "Surapet", "Tambaram", "Teynampet", "Tharamani",
    "T. Nagar", "Thirumangalam", "Thirumullaivoyal", "Thiruneermalai",
    "Thiruninravur", "Thiruvanmiyur", "Thiruvallur", "Tiruverkadu",
    "Thiruvotriyur", "Thuraipakkam", "Tirusulam", "Tiruvallikeni", "Tondiarpet",
    "United India Colony", "Vandalur", "Vadapalani", "Valasaravakkam",
    "Vallalar Nagar", "Vanagaram", "Velachery", "Velappanchavadi", "Villivakkam",
    "Virugambakkam", "Vyasarpadi", "Washermanpet", "West Mambalam",
]
def extract_location_from_text(text):
    text_lower = text.lower()
    abbrev_map = {
        "t nagar": "T. Nagar",
        "kkn": "K.K. Nagar",
        "besant": "Besant Nagar",
        "nungam": "Nungambakkam",
        "tambaram": "Tambaram",
        "cmbt": "Koyambedu",
        "koyambedu bus": "Koyambedu",
        "phoenix mall": "Velachery",
        "phoenix": "Velachery",
        "chennai airport": "Meenambakkam",
        "airport": "Meenambakkam",
        "central station": "Park Town",
        "central": "Park Town",
        "egmore station": "Egmore",
        "spencer": "Park Town",
        "tidel park": "Sholinganallur",
        "siruseri": "Sholinganallur",
        "omr": "Sholinganallur",
    }

    for short_form, full_form in abbrev_map.items():
        if short_form in text_lower:
            return full_form

    for area in sorted(CHENNAI_AREAS, key=len, reverse=True):
        if area.lower() in text_lower:
            return area

    return None


def analyze_accident_image(image_bytes, mime_type="image/jpeg"):
    if not GEMINI_API_KEY:
        return None, None

    prompt = (
        "You are an emergency response AI. Look at this accident scene image "
        "carefully. Based on visible damage, injuries, number of vehicles "
        "involved, and severity of the scene, classify the accident severity "
        "as exactly one of: High, Medium, or Low. High = life threatening "
        "injuries, unconscious people, severe crashes, severe vehicle damage, "
        "fire, vehicles badly crushed. Medium = visible injuries, moderate "
        "damage, people hurt but conscious. Low = minor damage, no visible "
        "injuries, small scratch, fender bender. Respond with ONLY the "
        "severity level and one sentence explanation."
    )

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            [
                prompt,
                {
                    "mime_type": mime_type,
                    "data": image_bytes,
                },
            ]
        )
        text = (response.text or "").strip()
        if not text:
            return None, None
        severity_match = re.search(r"\b(High|Medium|Low)\b", text, re.IGNORECASE)
        severity_token = severity_match.group(1).capitalize() if severity_match else None
        if severity_token not in {"High", "Medium", "Low"}:
            return None, text

        explanation_lower = text.lower()
        severe_damage_phrases = [
            "severe damage",
            "heavily damaged",
            "badly damaged",
            "crushed",
            "wrecked",
            "totaled",
            "multiple damaged vehicles",
            "front-end damage",
            "rear-end damage",
            "major collision",
            "major crash",
            "mangled",
            "twisted",
            "smashed",
            "severe collision",
            "heavy collision",
            "visible damage",
            "significant damage",
            "bad collision",
            "severe accident",
            "major accident",
            "two vehicles collided",
            "multiple vehicles",
            "vehicle damage",
            "car damage",
        ]
        if severity_token != "High" and any(phrase in explanation_lower for phrase in severe_damage_phrases):
            severity_token = "High"

        if severity_token != "High":
            crash_context_phrases = [
                "collision",
                "crash",
                "accident",
                "wreck",
                "damaged vehicle",
                "damaged cars",
                "debris",
                "airbag",
                "front of the car",
                "front of the vehicle",
                "two cars",
                "two vehicles",
                "multiple cars",
                "multiple vehicles",
            ]
            if any(phrase in explanation_lower for phrase in crash_context_phrases) and any(
                phrase in explanation_lower
                for phrase in ["damage", "damaged", "crushed", "smashed", "wrecked"]
            ):
                severity_token = "High"

        return severity_token, text
    except Exception:
        return None, None



st.set_page_config(page_title="RoadSoS", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #e6e6e6;
    }
    .roadsos-card {
        background: #151a21;
        border: 1px solid #2a2f3a;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0;
    }
    .roadsos-badge {
        display: inline-block;
        padding: 10px 16px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 18px;
        letter-spacing: 0.5px;
    }
    .roadsos-input-hero {
        font-size: 30px;
        font-weight: 800;
        line-height: 1.25;
        margin-bottom: 8px;
        color: #ffffff;
    }
    .roadsos-input-subhero {
        font-size: 18px;
        color: #d8dde6;
        margin-bottom: 10px;
    }
    .roadsos-input-panel {
        background: #151a21;
        border: 1px solid #2a2f3a;
        border-radius: 12px;
        padding: 14px 14px 8px 14px;
        margin-bottom: 10px;
    }
    .roadsos-side-hero {
        font-size: 22px;
        font-weight: 800;
        line-height: 1.3;
        color: #ffffff;
        margin-bottom: 12px;
    }
    .roadsos-photo-hero {
        font-size: 22px;
        font-weight: 800;
        line-height: 1.3;
        color: #ffffff;
        margin: 8px 0 10px 0;
    }
    .stButton > button[kind="primary"] {
        background-color: #E63946;
        border: 1px solid #E63946;
        color: #ffffff;
        font-weight: 800;
        font-size: 20px;
        min-height: 56px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")


@st.cache_resource
def load_models():
    model_path = MODELS_DIR / "severity_model.pkl"
    encoder_path = MODELS_DIR / "label_encoder.pkl"

    if not model_path.exists() or not encoder_path.exists():
        return None, None

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    return model, encoder


@st.cache_data
def load_osm_data(cache_buster=0):
    hospitals_path = DATA_DIR / "hospitals.pkl"
    police_path = DATA_DIR / "police.pkl"

    if cache_buster == 0 and hospitals_path.exists() and police_path.exists():
        hospitals = pd.read_pickle(hospitals_path)
        police = pd.read_pickle(police_path)
        return hospitals, police

    city = "Chennai, Tamil Nadu, India"
    hospitals = ox.features_from_place(city, tags={"amenity": "hospital"})
    hospitals = hospitals[["geometry", "name"]].copy()
    hospitals = hospitals.dropna(subset=["name"])
    hospitals["phone"] = hospitals.get("phone")
    hospitals["lat"] = hospitals.geometry.centroid.y
    hospitals["lon"] = hospitals.geometry.centroid.x
    hospitals = hospitals.reset_index(drop=True)

    police = ox.features_from_place(city, tags={"amenity": "police"})
    police = police[["geometry", "name"]].copy()
    police = police.dropna(subset=["name"])
    police["phone"] = police.get("phone")
    police["lat"] = police.geometry.centroid.y
    police["lon"] = police.geometry.centroid.x
    police = police.reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    hospitals.to_pickle(hospitals_path)
    police.to_pickle(police_path)

    return hospitals, police


@st.cache_data
def load_workshop_data(cache_buster=0):
    workshops_path = DATA_DIR / "workshops.pkl"

    if cache_buster == 0 and workshops_path.exists():
        return pd.read_pickle(workshops_path)

    city = "Chennai, Tamil Nadu, India"
    car_repair = ox.features_from_place(city, tags={"shop": "car_repair"})
    bike_repair = ox.features_from_place(city, tags={"shop": "motorcycle_repair"})

    workshops = pd.concat([car_repair, bike_repair], ignore_index=True)
    if "name" in workshops.columns:
        workshops = workshops[["geometry", "name"]].copy()
        workshops = workshops.dropna(subset=["name"])
    else:
        workshops = workshops[["geometry"]].copy()
        workshops["name"] = "Workshop"

    workshops["lat"] = workshops.geometry.centroid.y
    workshops["lon"] = workshops.geometry.centroid.x
    workshops = workshops.reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    workshops.to_pickle(workshops_path)

    return workshops


severity_keyword_rules = {
    "no_injury": [
        "no injury", "no injuries", "no one injured", "not injured", "no bleeding",
        "safe", "okay", "conscious", "alert", "no casualties",
    ],
    "high": [
        "dead", "died", "death", "unconscious", "not breathing", "can't breathe",
        "heavy bleeding", "blood everywhere", "bleeding badly", "trapped", "stuck inside",
        "crushed", "head injury", "skull fracture", "broken neck", "spine injury",
        "chest injury", "severe pain", "multiple fractures", "body not moving",
        "heart attack", "seizure", "fire", "vehicle burning", "explosion", "smoke from car",
        "car upside down", "rollover", "fell from bridge", "fell in ditch", "hit by lorry",
        "truck crash", "bus accident", "high speed crash", "major accident", "fatal accident",
        "person under vehicle", "pinned under bike", "emergency", "help immediately",
        "save me", "critical", "serious condition", "fainted",
    ],
    "medium": [
        "fracture", "broken arm", "broken leg", "injured", "hurt", "wounded", "broken",
        "pain", "bleeding", "deep cut", "unable to stand", "unable to walk", "collision",
        "airbag deployed", "passenger injured", "driver injured", "dizziness",
        "swelling", "neck pain", "back pain", "shoulder pain",
    ],
    "low": [
        "scratch", "small scratch", "minor injury", "slight pain", "little pain", "bruises",
        "small cut", "vehicle damage", "bumper damage", "mirror broken", "indicator broken",
        "slow speed accident", "parking accident", "tyre burst", "skid but safe", "fell but okay",
    ],
}

negation_words = set(
    [
        "no",
        "not",
        "without",
        "never",
        "cant",
        "can't",
        "dont",
        "don't",
        "illa",
        "illai",
        "illaye",
        "illaiy",
        "illaiyya",
    ]
)


def build_matcher(nlp, rules):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for label in ("no_injury", "high", "medium", "low"):
        phrases = rules.get(label, [])
        if phrases:
            patterns = [nlp.make_doc(p) for p in phrases]
            matcher.add(label.upper(), patterns)
    return matcher


def is_negated(span):
    start = max(0, span.start - 3)
    end = min(len(span.doc), span.end + 3)
    for tok in span.doc[start:end]:
        if tok.lower_ in negation_words or tok.dep_ == "neg":
            return True
    return False


def semantic_clean(text):
    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction="You are an emergency triage assistant. A panicked person described an accident. Rewrite their message as 1-2 clean sentences preserving ALL facts — injuries, vehicle damage, location, number of people. Do not add anything new. Just clean and complete the sentence."
        )
        response = model.generate_content(text)
        if response and response.text:
            return response.text.strip()
        return text
    except Exception:
        return text


def extract_accident_info(text, nlp):
    doc = nlp(text)
    matcher = build_matcher(nlp, severity_keyword_rules)
    matches = matcher(doc)
    score = 0
    matched = []

    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id].lower()
        span = doc[start:end]
        matched.append((label, span.text))
        if label == "no_injury" or is_negated(span):
            return {
                "severity": "Low",
                "confidence": 1.0,
                "matched": matched,
                "locations": [
                    ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC", "FAC"]
                ],
                "people_count": [
                    ent.text for ent in doc.ents if ent.label_ == "CARDINAL"
                ],
                "original_text": text,
                "needs_workshop": any(
                    kw in text.lower() for kw in [
                        "car damage", "vehicle damage", "bumper", "scratch", "dent",
                        "tyre", "tire", "mirror broken", "repair", "workshop", "tow"
                    ]
                ),
            }

    for label, span_text in matched:
        if label == "high":
            score += 3
        elif label == "medium":
            score += 2
        elif label == "low":
            score -= 1

    labels = [m[0] for m in matched]
    if "high" in labels:
        severity = "High"
    elif "medium" in labels:
        severity = "Medium"
    else:
        severity = "Low"

    confidence = 1 / (1 + math.exp(-0.8 * (score)))

    return {
        "severity": severity,
        "confidence": round(confidence, 3),
        "matched": matched,
        "locations": [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC", "FAC"]],
        "people_count": [ent.text for ent in doc.ents if ent.label_ == "CARDINAL"],
        "original_text": text,
        "needs_workshop": any(
            kw in text.lower() for kw in [
                "car damage", "vehicle damage", "bumper", "scratch", "dent",
                "tyre", "tire", "mirror broken", "repair", "workshop", "tow"
            ]
        ),
    }


def geocode_location(location_text):
    geolocator = Nominatim(user_agent="roadsos_app")
    queries = [
        f"{location_text}, Chennai, Tamil Nadu, India",
        f"{location_text}, Tamil Nadu, India",
        f"{location_text}, India",
    ]
    try:
        for query in queries:
            result = geolocator.geocode(query, timeout=10)
            if result is not None:
                return result.latitude, result.longitude
        return None
    except Exception:
        return None


def find_nearest_resources(accident_lat, accident_lon, hospitals_df, police_df):
    accident_location = (accident_lat, accident_lon)

    hospitals = hospitals_df.copy()
    police = police_df.copy()

    hospitals["distance_km"] = hospitals.apply(
        lambda row: geodesic(accident_location, (row["lat"], row["lon"])).km,
        axis=1,
    )

    police["distance_km"] = police.apply(
        lambda row: geodesic(accident_location, (row["lat"], row["lon"])).km,
        axis=1,
    )

    nearest_hospitals = hospitals.nsmallest(3, "distance_km")[
        ["name", "distance_km", "lat", "lon"]
    ]
    nearest_police = police.nsmallest(2, "distance_km")[
        ["name", "distance_km", "lat", "lon"]
    ]

    return nearest_hospitals, nearest_police


def build_emergency_map(accident_lat, accident_lon, hospitals_df, police_df):
    m = folium.Map(location=[accident_lat, accident_lon], zoom_start=13)

    folium.Marker(
        [accident_lat, accident_lon],
        popup="Accident Location",
        icon=folium.Icon(color="red", icon="exclamation-sign"),
    ).add_to(m)

    for _, row in hospitals_df.iterrows():
        folium.Marker(
            [row["lat"], row["lon"]],
            popup=f"Hospital: {row['name']} ({row['distance_km']:.2f} km)",
            icon=folium.Icon(color="blue", icon="plus-sign"),
        ).add_to(m)

    for _, row in police_df.iterrows():
        folium.Marker(
            [row["lat"], row["lon"]],
            popup=f"Police: {row['name']} ({row['distance_km']:.2f} km)",
            icon=folium.Icon(color="darkblue", icon="home"),
        ).add_to(m)

    return m


@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")


def transcribe_audio_bytes(audio_bytes):
    if not audio_bytes or len(audio_bytes) < 1000:
        return ""
    model = load_whisper_model()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(audio_bytes)
        temp_path = temp_file.name
    try:
        result = model.transcribe(temp_path, fp16=False)
        return result.get("text", "").strip()
    except Exception:
        return ""
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


st.title("RoadSoS — AI Emergency Response System")

with st.sidebar:
    refresh_osm = st.button("Refresh OSM data (slow)")

cache_buster = int(refresh_osm)

try:
    hospitals, police = load_osm_data(cache_buster=cache_buster)
except Exception:
    hospitals, police = None, None

try:
    workshops = load_workshop_data(cache_buster=cache_buster)
except Exception:
    workshops = None

nlp = load_nlp()
model, encoder = load_models()


(tab_emergency, tab_blackspot, tab_about) = st.tabs(
    ["Emergency Response", "Blackspot Map", "About"]
)

with tab_emergency:
    st.subheader("Emergency Response")

    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "resolved_coords" not in st.session_state:
        st.session_state.resolved_coords = None
    if "location_source" not in st.session_state:
        st.session_state.location_source = None
    if "request_gps" not in st.session_state:
        st.session_state.request_gps = False

    st.markdown(
        """
        <div class="roadsos-card">
            🚑 <a href="tel:108">Ambulance - 108</a>
            &nbsp;&nbsp;🚔 <a href="tel:100">Police - 100</a>
            &nbsp;&nbsp;🔥 <a href="tel:101">Fire - 101</a>
            &nbsp;&nbsp;🏥 <a href="tel:104">Health Helpline - 104</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="roadsos-input-hero">Don&apos;t Panic. Take a deep breath and tell us what happened.</div>
        <div class="roadsos-input-subhero">RoadSoS is here to help.</div>
        """,
        unsafe_allow_html=True,
    )

    input_col, voice_col = st.columns([2, 1], vertical_alignment="top")

    with input_col:
        st.markdown("<div class='roadsos-input-panel'>", unsafe_allow_html=True)
        text_input = st.text_area(
            "Describe the accident",
            placeholder="Accident near Anna Salai, 2 people injured",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    audio_text = ""
    with voice_col:
        st.markdown(
            """
            <div class="roadsos-input-panel">
                <div class="roadsos-side-hero">Can&apos;t type ?? no problem, record your message.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if MIC_AVAILABLE:
            audio_data = mic_recorder(
                start_prompt="Start recording", stop_prompt="Stop recording", format="wav"
            )
            if audio_data and isinstance(audio_data, dict) and audio_data.get("bytes"):
                with st.spinner("🎙️ Transcribing your voice..."):
                    raw_transcription = transcribe_audio_bytes(audio_data["bytes"])
                audio_text = st.text_area(
                    "Transcribed audio (edit if needed)",
                    value=raw_transcription,
                    key="audio_text_edit",
                    height=80,
                )
        else:
            st.info("Mic recorder not available. Install streamlit-mic-recorder to enable it.")

    st.markdown(
        '<div class="roadsos-photo-hero">Have a photo? Upload it to help us assess faster.</div>',
        unsafe_allow_html=True,
    )
    image_file = None
    if GEMINI_API_KEY:
        image_file = st.file_uploader(
            "Upload accident photo (optional)", type=["jpg", "jpeg", "png"]
        )

    analyze = st.button("🚨 ANALYZE ACCIDENT / HELP", type="primary", use_container_width=True)

    if analyze:
        st.session_state.resolved_coords = None
        st.session_state.location_source = None
        st.session_state.request_gps = False
        input_text = text_input.strip() or audio_text.strip()
        has_image = image_file is not None
        if not input_text and not has_image:
            st.warning("Please provide accident text, record audio, or upload an image.")
        else:
            cleaned_text = input_text
            if input_text:
                with st.spinner("🧠 Understanding your description..."):
                    cleaned_text = semantic_clean(input_text) if GEMINI_API_KEY else input_text
                    info = extract_accident_info(cleaned_text, nlp)
            else:
                info = {
                    "severity": "Low",
                    "needs_workshop": False,
                    "locations": [],
                    "people_count": [],
                    "original_text": "",
                }
            severity_colors = {
                "High": "#e74c3c",
                "Medium": "#f39c12",
                "Low": "#2ecc71",
            }
            text_severity = info["severity"]

            image_severity = None
            image_explanation = None
            if has_image:
                image_severity, image_explanation = analyze_accident_image(
                    image_file.getvalue(), image_file.type or "image/jpeg"
                )

            image_workshop_hit = image_severity in {"High", "Medium"}
            if image_explanation:
                explanation_lower = image_explanation.lower()
                image_workshop_hit = any(
                    phrase in explanation_lower
                    for phrase in [
                        "damage",
                        "crushed",
                        "broken",
                        "repair",
                        "workshop",
                        "tow",
                        "towing",
                        "vehicle",
                    ]
                )

            severity_rank = {"Low": 0, "Medium": 1, "High": 2}
            candidates = []
            if text_severity in severity_rank:
                candidates.append(text_severity)
            if image_severity in severity_rank:
                candidates.append(image_severity)
            final_severity = (
                max(candidates, key=lambda s: severity_rank.get(s, 0))
                if candidates
                else "Low"
            )

            location_text_display = (
                extract_location_from_text(cleaned_text)
                or (info["locations"][0] if info["locations"] else None)
            )
            people_text_display = ", ".join(info["people_count"]) if info["people_count"] else None

            st.session_state.analysis_result = {
                "severity": final_severity,
                "severity_color": severity_colors.get(final_severity, "#e6e6e6"),
                "image_severity": image_severity,
                "image_explanation": image_explanation,
                "location_text_display": location_text_display,
                "people_text_display": people_text_display,
                "needs_workshop": info.get("needs_workshop", False),
                "image_workshop_hit": image_workshop_hit,
                "location_text": location_text_display or "",
            }

    analysis = st.session_state.analysis_result

    if analysis:
        st.markdown(
            f"""
            <div class="roadsos-card">
                <div class="roadsos-badge" style="background:{analysis['severity_color']}; color:#0e1117;">
                    Severity: {analysis['severity']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if analysis["image_severity"]:
            image_color = severity_colors.get(analysis["image_severity"], "#e6e6e6")
            st.markdown(
                f"""
                <div class="roadsos-card">
                    <div class="roadsos-badge" style="background:{image_color}; color:#0e1117;">
                        📸 Image Severity: {analysis['image_severity']}
                    </div>
                    <div style="margin-top: 8px;">{analysis['image_explanation']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        location_line = (
            f"📍 **Location detected:** {analysis['location_text_display']}"
            if analysis["location_text_display"]
            else "📍 **Location:** Not detected"
        )
        people_line = (
            f"👥 **People involved:** {analysis['people_text_display']}"
            if analysis["people_text_display"]
            else "👥 **People involved:** Not mentioned"
        )

        st.markdown(
            f"""
            <div class="roadsos-card" style="border: 1px solid #2a2f3a;">
                <div>{location_line}</div>
                <div>{people_line}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if analysis["location_text_display"] and st.session_state.resolved_coords is None:
            st.write("Attempting to geocode:", analysis["location_text_display"])
            st.session_state.resolved_coords = geocode_location(analysis["location_text_display"])
            if st.session_state.resolved_coords is not None:
                st.session_state.location_source = "text"

        if st.session_state.resolved_coords is None:
            st.info("We couldn't detect your location. Type your area name clearly or use your current location:")
            gps_trigger = st.button("Use current location (GPS)", key="use_current_location_button")
            if gps_trigger:
                st.session_state.request_gps = True

            if st.session_state.request_gps:
                if GEOLOCATION_AVAILABLE:
                    st.caption("Please allow browser location access to continue.")
                    gps_data = streamlit_geolocation()
                    if gps_data and gps_data.get("latitude") is not None and gps_data.get("longitude") is not None:
                        st.session_state.resolved_coords = (
                            gps_data["latitude"],
                            gps_data["longitude"],
                        )
                        st.session_state.location_source = "gps"
                        st.session_state.request_gps = False
                        st.success("Using your current location.")
                else:
                    st.warning("GPS support package not found. Install `streamlit-geolocation` to enable this feature.")

            retry_location = st.text_input(
                "Retry location",
                key="retry_location_input",
                value=analysis["location_text"] or "",
            )
            retry = st.button("Retry location", key="retry_location_button")
            if retry and retry_location.strip():
                st.session_state.resolved_coords = geocode_location(retry_location.strip())
                if st.session_state.resolved_coords is not None:
                    st.session_state.location_source = "manual"

        coords = st.session_state.resolved_coords

        if coords is None:
            st.error("Location not found. Please provide a clearer area name.")
        elif hospitals is None or police is None:
            st.error("OSM data not available. Use refresh or check your connection.")
        else:
            nearest_hospitals, nearest_police = find_nearest_resources(
                coords[0], coords[1], hospitals, police
            )

            severity = analysis["severity"]
            needs_workshop = (
                analysis["needs_workshop"]
                or analysis["image_workshop_hit"]
                or analysis["severity"] == "Low"
            )

            if severity == "High":
                st.markdown("**Ambulance**")
                st.link_button("Call Ambulance 108", "tel:108")

            if severity in ["High", "Medium"]:
                st.markdown("**Nearest hospitals**")
                for _, row in nearest_hospitals.iterrows():
                    st.markdown(f"- {row['name']} — {row['distance_km']:.2f} km")
                    maps_url = (
                        "https://www.google.com/maps/dir/?api=1"
                        f"&destination={row['lat']},{row['lon']}"
                    )
                    st.link_button("Open in Google Maps", maps_url)

            st.markdown("**Nearest police stations**")
            for _, row in nearest_police.iterrows():
                st.markdown(f"- {row['name']} — {row['distance_km']:.2f} km")
                maps_url = (
                    "https://www.google.com/maps/dir/?api=1"
                    f"&destination={row['lat']},{row['lon']}"
                )
                st.link_button("Open in Google Maps", maps_url)

            if needs_workshop:
                if workshops is None or workshops.empty:
                    st.warning("Workshop data not available right now.")
                else:
                    workshop_points = workshops.copy()
                    workshop_points["distance_km"] = workshop_points.apply(
                        lambda row: geodesic(
                            (coords[0], coords[1]), (row["lat"], row["lon"])
                        ).km,
                        axis=1,
                    )
                    nearest_workshops = workshop_points.nsmallest(3, "distance_km")

                    st.markdown("**Nearest workshops**")
                    for _, row in nearest_workshops.iterrows():
                        st.markdown(f"- {row['name']} — {row['distance_km']:.2f} km")
                        maps_url = (
                            "https://www.google.com/maps/dir/?api=1"
                            f"&destination={row['lat']},{row['lon']}"
                        )
                        st.link_button("Open in Google Maps", maps_url)

            if severity in ["High", "Medium"]:
                emergency_map = build_emergency_map(
                    coords[0], coords[1], nearest_hospitals, nearest_police
                )
                st.iframe(emergency_map._repr_html_(), height=600)

with tab_blackspot:
    st.subheader("Chennai Blackspot Map")
    blackspot_path = DATA_DIR / "blackspot_map.html"
    if blackspot_path.exists():
        blackspot_html = blackspot_path.read_text(encoding="utf-8")
        st.iframe(blackspot_html, height=600)
    else:
        st.warning("blackspot_map.html not found. Run day5_blackspot.ipynb first.")

with tab_about:
    st.subheader("About RoadSoS")
    st.write(
        "RoadSoS is an AI-powered emergency accident triage and response system. "
        "It combines NLP, ML, and real map data to help route help quickly and visualize risk zones."
    )
    st.markdown(
        "**Pipeline**: Voice or text input → Whisper transcription → spaCy extraction → "
        "Severity prediction → Nearest hospital/police search → Interactive map"
    )
