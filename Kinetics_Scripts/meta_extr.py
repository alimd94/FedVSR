
import os
import cv2
import pandas as pd
from tqdm import tqdm
import re

# Load CSV file with annotations
f = pd.read_csv("path to annotations/train.csv")

# Regex pattern to extract ID, start, and end
pattern = r"^(.*?)_(\d+)_(\d+)\.mp4$"

# Directory containing videos
video_dir = "path to k400/train"

# Output metadata
metadata = []

for video_file in tqdm(os.listdir(video_dir)):
    if video_file.endswith((".mp4", ".avi", ".mov")):
        match = re.match(pattern, video_file)
        video_id, start, end = match.groups()
        video_path = os.path.join(video_dir, video_file)
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {video_file}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps else 0
            label_row = f[f['youtube_id'] == video_id]
            label = label_row['label'].values[0] if not label_row.empty else "Unknown"
            metadata.append({
                "video": video_file,
                "fps": fps,
                "resolution": f"{int(width)}x{int(height)}",
                "length_seconds": duration,
                "label": label
            })
            cap.release()
        except Exception as e:
            print(f"Error processing file {video_file}: {e}")

# Save metadata to a CSV file
metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv("video_metadata_with_labels.csv", index=False)

print("Metadata saved as CSV successfully.")
