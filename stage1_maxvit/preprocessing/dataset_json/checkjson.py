import json
from pathlib import Path

json_path = Path(
    r"FaceForensics++.json"
)

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

root_key = list(data.keys())[0]
print("Root key:", root_key)

for sub_key in data[root_key].keys():
    print("Sub key:", sub_key)

    for split in ["train", "val", "test"]:
        total_videos = 0
        total_frames = 0

        if split not in data[root_key][sub_key]:
            continue

        for comp, videos in data[root_key][sub_key][split].items():
            total_videos += len(videos)
            for vid, info in videos.items():
                total_frames += len(info["frames"])

        print(f"  {split}: videos={total_videos}, frames={total_frames}")