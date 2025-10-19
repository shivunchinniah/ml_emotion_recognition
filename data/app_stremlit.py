import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io

# Emotion mapping
emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

@st.cache_data
def load_data():
    """Loads and returns the dataframe."""
    return pd.read_csv('fer2013.csv')

def pixels_to_image(pixels_str):
    """Converts pixel string to a PIL Image."""
    pixels = np.array(pixels_str.split(), 'uint8')
    image = pixels.reshape(48, 48)
    return Image.fromarray(image, 'L')

# --- Main App ---
df = load_data()
total_images = len(df)

st.title("FER-2013 Manual Image Inspector")
st.markdown("Use the controls below to inspect images and their assigned emotion labels for your report.")

# Initialize or update the current image index in the session state
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

# Navigation Buttons
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("Previous Image", disabled=st.session_state.current_index == 0):
        st.session_state.current_index -= 1
        st.rerun()

with col3:
    if st.button("Next Image", disabled=st.session_state.current_index == total_images - 1):
        st.session_state.current_index += 1
        st.rerun()

current_index = st.session_state.current_index

# Get data for the current image
row = df.iloc[current_index]
image_id = current_index
emotion_label = emotion_map[row['emotion']]
usage = row['Usage']
img = pixels_to_image(row['pixels'])

st.subheader(f"Image {current_index + 1} of {total_images} (ID: {image_id})")
st.metric(label="Assigned Emotion Label", value=emotion_label)
st.info(f"Dataset Usage: {usage}")

# Display the image, scaled up for inspection
st.image(img, caption=f"ID: {image_id}, Label: {emotion_label}", width=250)
st.markdown("---")

st.subheader("Inspection Area")
st.write(f"**Image ID:** `{image_id}` | **Label:** `{emotion_label}`")
st.text_area("Your Notes (Good/Bad Example, Observations):", key=f"notes_{image_id}", height=100)

# To run this Streamlit app:
# streamlit run app_streamlit.py