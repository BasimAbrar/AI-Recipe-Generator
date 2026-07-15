import streamlit as st
from ultralytics import YOLO
import pandas as pd
import numpy as np
from PIL import Image
import tempfile, os

st.set_page_config(
    page_title="Recipe Generator",
    page_icon="🍽️",
    layout="centered"
)

@st.cache_resource
def load_models():
    m1 = YOLO("/kaggle/working/recipe_generator/runs/train/weights/best.pt")
    m2 = YOLO("yolov8n.pt")
    return m1, m2

@st.cache_data
def load_recipes():
    df = pd.read_csv("/kaggle/input/datasets/hugodarwood/epirecipes/epi_r.csv")
    meta = ["rating","calories","protein","fat","sodium","title"]
    ingredient_cols = [c for c in df.columns if c not in meta]
    df[ingredient_cols] = df[ingredient_cols].fillna(0)
    return df, ingredient_cols

model_lvis, model_coco = load_models()
df, ingredient_cols    = load_recipes()

COCO_FOOD_IDS = {
    46:"banana", 47:"apple", 48:"sandwich", 49:"orange",
    50:"broccoli", 51:"carrot", 52:"hot dog", 53:"pizza",
    54:"donut", 55:"cake"
}

def detect_ingredients(image_path, conf_lvis=0.25, conf_coco=0.30):
    ingredients = set()
    r1 = model_lvis(image_path, verbose=False, conf=conf_lvis)
    if r1[0].boxes is not None:
        for cls in r1[0].boxes.cls:
            name = model_lvis.names[int(cls)].split("/")[0].strip().lower()
            ingredients.add(name)
    r2 = model_coco(image_path, verbose=False, conf=conf_coco)
    if r2[0].boxes is not None:
        for cls in r2[0].boxes.cls:
            cid = int(cls)
            if cid in COCO_FOOD_IDS:
                ingredients.add(COCO_FOOD_IDS[cid])
    return sorted(ingredients)

def find_recipes(detected, top_n=5):
    if not detected:
        return pd.DataFrame()
    valid = [i for i in detected if i in ingredient_cols]
    if not valid:
        return pd.DataFrame()
    scores   = df[valid].sum(axis=1)
    total    = df[ingredient_cols].sum(axis=1)
    bonus    = scores / total.replace(0, 1)
    combined = scores + (0.3 * bonus)
    out = df[["title"] + valid].copy()
    out["match_count"] = scores.values
    out["score"]       = combined.round(3)
    out = out[out["match_count"] > 0]
    out = out.sort_values("score", ascending=False)
    out = out.drop_duplicates(subset="title")
    return out.head(top_n).reset_index(drop=True)

st.title("🍽️ AI Recipe Generator")
st.markdown("Upload a photo of your ingredients and get instant recipe suggestions.")
st.divider()

uploaded = st.file_uploader(
    "📸 Upload an ingredient photo",
    type=["jpg","jpeg","png"]
)

conf_thresh = st.slider("Detection confidence", 0.10, 0.80, 0.25, 0.05)

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    col1, col2 = st.columns(2)
    with col1:
        st.image(tmp_path, caption="Uploaded image", use_column_width=True)

    with st.spinner("🔍 Detecting ingredients..."):
        detected = detect_ingredients(tmp_path, conf_lvis=conf_thresh, conf_coco=conf_thresh)

    with col2:
        st.markdown("### 🥕 Detected Ingredients")
        if detected:
            for ing in detected:
                st.success(f" {ing}")
        else:
            st.warning("No ingredients detected. Try a clearer photo or lower the confidence slider.")

    st.divider()

    if detected:
        with st.spinner("🍳 Finding best recipes..."):
            matches = find_recipes(detected, top_n=5)

        if matches.empty:
            st.warning("Ingredients detected but no matching recipes found. Try different ingredients.")
        else:
            st.markdown("### 🍽️ Top Recipe Suggestions")
            for i, row in matches.iterrows():
                matched_cols = [c for c in row.index
                                if c not in ["title","match_count","score"]
                                and row[c] == 1]
                with st.expander(f"{'🥇' if i==0 else '🥈' if i==1 else '🥉' if i==2 else '▪️'} {row['title']}"):
                    st.markdown(f"**Matched ingredients:** {', '.join(matched_cols)}")
                    st.markdown(f"**Match count:** {int(row['match_count'])}  |  **Score:** {row['score']}")
                    recipe_row = df[df["title"] == row["title"]].iloc[0]
                    all_ings = [c for c in ingredient_cols if recipe_row[c] == 1.0]
                    st.markdown(f"**All recipe ingredients:** {', '.join(all_ings[:20])}")

    os.unlink(tmp_path)
