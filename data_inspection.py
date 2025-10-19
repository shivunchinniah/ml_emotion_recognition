import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob

# --- Configuration ---
FER2013_MAIN_CSV = 'data/fer2013.csv' 
CSV_DIR = 'good_bad/'
SAMPLES_PER_STATUS = 4 # The number of samples (Good/Bad) per emotion class

# --- Helper Function for FER2013 Pixel Processing ---
def fer_pixels_to_array(pixel_string):
    """Converts a space-separated string of pixels into a 48x48 numpy array (or inferred size)."""
    data = np.array(pixel_string.split(), 'float32')
    size = int(np.sqrt(len(data)))
    if size * size != len(data):
        raise ValueError(f"Pixel string length ({len(data)}) is not a perfect square. Check FER2013 data integrity.")
    data = data.reshape(size, size)
    return data

# --- Loading and Combining Custom Labels ---
def load_custom_labels(csv_directory):
    # ... (function body as previously defined) ...
    print(f"Loading custom label CSVs from: {csv_directory}")
    
    all_files = glob(os.path.join(csv_directory, '*.csv'))
    if not all_files:
        print("No custom label CSV files found. Please check the CSV_DIR path.")
        return pd.DataFrame()

    df_list = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename, names=['Original ID', 'Emotion Label', 'Selection Status'], header=None)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading file {filename}: {e}")

    if not df_list:
        return pd.DataFrame()

    combined_df = pd.concat(df_list, ignore_index=True).drop_duplicates()
    combined_df.dropna(subset=['Original ID', 'Emotion Label', 'Selection Status'], inplace=True)
    combined_df['Original ID'] = combined_df['Original ID'].astype(int) 
    return combined_df

# --- Sampling Function ---
def sample_data(combined_df, samples_per_status):
    # ... (function body as previously defined) ...
    print("Sampling data...")
    sampled_data = []
    
    emotion_classes = combined_df['Emotion Label'].unique()
    
    for emotion in emotion_classes:
        emotion_df = combined_df[combined_df['Emotion Label'] == emotion]
        
        for status in ['Good', 'Bad']:
            status_df = emotion_df[emotion_df['Selection Status'] == status]
            
            if len(status_df) >= samples_per_status:
                sample = status_df.sample(n=samples_per_status, random_state=42)
            else:
                print(f"Warning: Only {len(status_df)} '{status}' samples for '{emotion}'. Taking all.")
                sample = status_df
            
            sampled_data.append(sample)
            
    return pd.concat(sampled_data, ignore_index=True)

# --- Plotting Function ---
def plot_pixel_grid(sampled_df, fer2013_main_csv, samples_per_status):
    # ... (function body as previously defined) ...
    print(f"Loading main FER2013 data from {fer2013_main_csv} for pixel lookup...")
    try:
        main_fer_df = pd.read_csv(fer2013_main_csv, usecols=['pixels'])
    except Exception as e:
        print(f"Error loading FER2013 main CSV: {e}")
        return

    # Determine figure layout
    emotion_classes = sorted(sampled_df['Emotion Label'].unique())
    statuses = ['Good', 'Bad']
    num_rows = len(emotion_classes)
    num_cols = len(statuses) * samples_per_status
    
    fig, axes = plt.subplots(
        nrows=num_rows, 
        ncols=num_cols, 
        figsize=(18, 4 * num_rows)
    )
    
    if num_rows == 1:
        axes = np.array([axes]).reshape(1, num_cols)
        
    print(f"Generating figure with {num_rows} rows and {num_cols} columns...")
    
    for i, emotion in enumerate(emotion_classes):
        row_df = sampled_df[sampled_df['Emotion Label'] == emotion]
        
        for j, status in enumerate(statuses):
            col_start_index = j * samples_per_status
            status_df = row_df[row_df['Selection Status'] == status]
            
            for k, (index, row) in enumerate(status_df.iterrows()):
                if k >= samples_per_status:
                    break
                    
                col_index = col_start_index + k
                ax = axes[i, col_index]
                original_id = row['Original ID']
                
                try:
                    pixel_string = main_fer_df.loc[original_id, 'pixels']
                    img_array = fer_pixels_to_array(pixel_string)
                    
                    ax.imshow(img_array, cmap='gray')
                    ax.set_title(f"{status}\nID: {original_id}", fontsize=8)
                    ax.axis('off')
                    
                except Exception:
                    ax.text(0.5, 0.5, f"ID {original_id}\nData Error", 
                            ha='center', va='center', color='red', fontsize=10)
                    ax.set_title(f"{status}", fontsize=8)
                    ax.axis('off')
                
        # Add row title
        fig.text(0.01, 
                 (num_rows - i - 0.5) / num_rows, 
                 f"{emotion}", 
                 ha='left', va='center', fontsize=14, weight='bold')


    plt.tight_layout(rect=[0.03, 0, 1, 1])
    plt.suptitle(f"FER2013 Data Sampling: {samples_per_status} Good/Bad Examples Per Class", 
                 y=1.02, fontsize=16, weight='bold')
    
    output_filename = 'sampled_fer2013_pixels.png'
    plt.savefig(output_filename)
    print(f"\nFigure saved as '{output_filename}'")

# =================================================================
# --- Main Execution Block (The missing piece) ---
# =================================================================
if __name__ == '__main__':
    
    # 1. Load custom labels
    custom_labels_df = load_custom_labels(CSV_DIR)
    
    if custom_labels_df.empty:
        print("Stopping execution.")
    else:
        # 2. Sample the required number of 'Good' and 'Bad' images per class
        sampled_df = sample_data(custom_labels_df, SAMPLES_PER_STATUS)
        
        if sampled_df.empty:
            print("Stopping execution because no samples were generated.")
        else:
            # 3. Generate and display the image grid using pixel lookups
            plot_pixel_grid(sampled_df, FER2013_MAIN_CSV, SAMPLES_PER_STATUS)