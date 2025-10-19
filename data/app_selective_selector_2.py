import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io

# --- Configuration ---
SAMPLES_PER_CLASS = 10
# Emotion mapping for FER-2013: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# --- Data Loading and Processing Functions ---

@st.cache_data
def load_data():
    """Loads and returns the dataframe."""
    # NOTE: Ensure 'fer2013.csv' is in the same directory as this script.
    try:
        df = pd.read_csv('fer2013.csv')
        return df
    except FileNotFoundError:
        st.error("Error: 'fer2013.csv' not found. Please place the file in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()


def pixels_to_image(pixels_str):
    """Converts pixel string to a PIL Image."""
    try:
        pixels = np.array(pixels_str.split(), 'uint8')
        image = pixels.reshape(48, 48)
        # 'L' mode is for grayscale
        return Image.fromarray(image, 'L')
    except:
        # Handle rows where pixel data might be corrupt or missing
        return Image.new('L', (48, 48), color='black') # Return a black image


def create_sample_df(df, samples_per_class):
    """Samples 'samples_per_class' images for each emotion label."""
    sampled_rows = []
    
    # Get a new random seed for each reshuffle
    random_seed = np.random.randint(0, 100000)

    for emotion_id in emotion_map.keys():
        class_df = df[df['emotion'] == emotion_id]
        
        # Use .sample(n, replace=False) to get unique random samples
        if len(class_df) >= samples_per_class:
            # Use random_state for reproducible samples within one run until reshuffle
            sampled_rows.append(class_df.sample(samples_per_class, random_state=random_seed))
        else:
            # For small classes (like Disgust), take all available samples
            sampled_rows.append(class_df)

    # Combine and reset index for easier iteration
    return pd.concat(sampled_rows).reset_index(drop=True)

# --- Streamlit Application Logic ---

df_full = load_data() # FIX: Load the full dataframe immediately
total_images = len(df_full)

st.title("FER-2013 Selective Sample Inspector 🧐")
st.markdown("Inspect **10 random samples** for each of the 7 emotion classes. Mark them as 'Good' or 'Bad' examples for your report.")

# --- Reshuffle Button (Always Visible) ---
# Create the button first
reshuffle_clicked = st.button("Reshuffle Samples 🎲 (Loads New Set, Keeps Logged Data)", type="primary")

# Initialize inspection_results if it doesn't exist
if 'inspection_results' not in st.session_state:
    st.session_state.inspection_results = {}
    
# Initialize or reset the sampling dataframe
# Pass df_full to the sampling function
if 'sampled_df' not in st.session_state or reshuffle_clicked:
    st.session_state.sampled_df = create_sample_df(df_full, SAMPLES_PER_CLASS)
    # ------------------------------------------------------------------
    # FIX: REMOVED the line below to prevent clearing of the logged data
    # st.session_state.inspection_results = {} # This line was removed
    # ------------------------------------------------------------------
    st.toast("New random samples loaded!")

# Helper to load data from state
sampled_df = st.session_state.sampled_df
inspection_results = st.session_state.inspection_results

# Create 3 columns for layout
cols = st.columns(3)

# Iterate through all unique emotion classes and display the samples
for i, (emotion_id, emotion_name) in enumerate(emotion_map.items()):
    # Filter the sampled data for the current emotion
    class_samples = sampled_df[sampled_df['emotion'] == emotion_id]
    
    # Use columns to display images in a grid format
    with cols[i % 3]: # Cycle through the 3 columns
        st.subheader(f"Class: {emotion_name}")
        st.markdown(f"**Total Samples:** {len(class_samples)}")
        
        for index, row in class_samples.iterrows():
            
            # --- Finding the Original ID ---
            # Search the full dataframe for the matching 'pixels' string
            # Use iloc to get the index (original ID) of the first match
            # This is the line that was corrected to use the now-defined df_full
            # Note: This lookup might be slow if the full dataframe is very large.
            original_id = df_full[df_full['pixels'] == row['pixels']].index[0] 
            
            unique_key = f"{original_id}_{emotion_name}"
            
            # Convert pixels to image
            img = pixels_to_image(row['pixels'])

            # Display image (scaled up)
            st.image(img, caption=f"Original ID: {original_id}", width=120)

            # Check for existing result or set to 'Unmarked'
            current_status = inspection_results.get(unique_key, 'Unmarked')
            
            # Use three columns for the buttons
            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])

            # Use a button click to update the inspection_results dictionary
            # Button 1: Good Example (Tick)
            # The key logic ensures that even if the image was marked in a previous shuffle,
            # the status is loaded correctly from inspection_results.
            if btn_col1.button("✅ Good", key=f"good_{unique_key}", disabled=(current_status == 'Good')):
                st.session_state.inspection_results[unique_key] = 'Good'
                st.rerun()

            # Button 2: Bad Example (X)
            if btn_col2.button("❌ Bad", key=f"bad_{unique_key}", disabled=(current_status == 'Bad')):
                st.session_state.inspection_results[unique_key] = 'Bad'
                st.rerun()

            # Display the current selection status
            with btn_col3:
                if current_status == 'Good':
                    st.success("Selected: Good")
                elif current_status == 'Bad':
                    st.error("Selected: Bad")
                else:
                    st.warning("Status: Unmarked")
            
            st.markdown("---")


# --- Final Report Section (Sidebar) ---
st.sidebar.title("Inspection Summary")

# Create a final report dataframe from the inspection results
report_data = []
for key, status in st.session_state.inspection_results.items():
    if status != 'Unmarked':
        original_id, emotion_name = key.split('_')
        report_data.append({
            "Original ID": int(original_id),
            "Emotion Label": emotion_name,
            "Selection Status": status
        })

if report_data:
    # Ensure only unique original IDs are logged in case of duplicate sampling, though
    # create_sample_df should ensure uniqueness within a single run.
    report_df = pd.DataFrame(report_data).drop_duplicates(subset=['Original ID', 'Emotion Label'])
    
    # Show counts of Good/Bad examples
    st.sidebar.subheader("Counts:")
    status_counts = report_df['Selection Status'].value_counts().to_frame()
    st.sidebar.dataframe(status_counts)
    
    st.sidebar.subheader("Details:")
    # Display the final, consolidated report
    st.sidebar.dataframe(report_df, use_container_width=True)
    
    # Download link for the final report
    csv_report = report_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download Report CSV",
        data=csv_report,
        file_name='fer2013_cumulative_inspection_report.csv',
        mime='text/csv',
    )
else:
    st.sidebar.info("No images have been marked yet.")