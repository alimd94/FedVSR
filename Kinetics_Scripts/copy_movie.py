import shutil
import os
import pandas as pd
import subprocess


dst_dir = "path to copy"
root_dir = "path to k400/train"


f = pd.read_csv("video_metadata_with_labels.csv")
diverse_classes = [
    # Nature-Related Activities
    "feeding birds",
    "planting trees",
    "rock climbing",
    "sailing",
    "bee keeping",
    # Sports
    "archery",
    "bungee jumping",
    "surfing water",
    "skiing crosscountry",
    "playing paintball",
    "snowboarding",
    # Music and Instruments
    "playing piano",
    "playing bagpipes",
    "playing didgeridoo",
    "playing violin",
    "playing accordion",
    # Dance and Performing Arts
    "dancing gangnam style",
    "marching",
    "tap dancing",
    "robot dancing",
    "yoga",
    # Food Preparation and Eating
    "baking cookies",
    "peeling potatoes",
    "breading or breadcrumbing",
    "making pizza",
    "making tea",
    # Interaction with Animals
    "feeding fish",
    "riding elephant",
    "milking cow",
    "petting cat",
    "feeding goats",
    # Creative or Object-Related Activities
    "blowing glass",
    "carving pumpkin",
    "braiding hair",
    "building cabinet",
    "folding paper",
    # Miscellaneous
    "snowmobiling",
    "spray painting",
    "swinging on something",
    "decorating the christmas tree"
]
metadata = []
i = 0
for l in diverse_classes:
    for v in f[(f['resolution']=='1280x720') & (f['fps']==25)&(f['length_seconds'] >= 4) & (f['label']==l)].head(6).video:
        src = os.path.join(root_dir, v)
        dst = os.path.join(dst_dir, f'{i:03}')
        os.makedirs(dst, exist_ok=True)
        dst = os.path.join(dst, v)

        video_dir = os.path.dirname(dst)
        
        command = [
            "ffmpeg",
            "-i", src,           # Input video file
            "-frames:v", "100",         # Extract only the first 100 frames
            "-vf", "fps=25",            # Match 25 FPS
            os.path.join(video_dir, "%08d.png")  # Naming pattern for output
        ]

        # Run the command
        subprocess.run(command, check=True)
        i+=1
        metadata.append({
                "video": dst.split("/")[-2],
                "label": l
            })

metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv("video_possible_with_labels.csv", index=False)
