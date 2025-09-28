# check_almond.py
# ------------------------------------------
# System for getting VLM data on almond images.
# Checks if a constant image contains almonds, and counts them.
# Output: prints ONLY "<true|false> <count>" to stdout.

import os
import sys
import json
import base64
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from tqdm import tqdm
import re
import pandas as pd
import matplotlib.pyplot as plt

load_dotenv()

# -------- Config --------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "moondream:1.8b")
TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT_SEC", "60"))
MAX_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "1024"))

IMAGE_PATH = "test.jpg" # no longer used as iterates through a directory of images now.


def encode_image_b64_raw(path: str) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if MAX_SIDE > 0 and max(w, h) > MAX_SIDE:
            scale = MAX_SIDE / float(max(w, h))
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_ollama_with_images(image_b64: str, model: str) -> str:
    url = f"{OLLAMA_HOST.rstrip('/')}/api/chat"
    headers = {"Content-Type": "application/json"}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": "Answer strictly in this JSON format: {\"contains_almond\": true/false, \"count\": integer}",
            },
            {
                "role": "user",
                "content": (
                    "Does this image contain almonds (the edible nut)? "
                    "If yes, estimate how many. "
                    "Respond ONLY with JSON: {\"contains_almond\": true/false, \"count\": integer}"
                ),
                "images": [image_b64],
            },
        ],
    }

    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["message"]["content"]


def parse_result(reply: str) -> tuple[bool, int]:
    try:
        data = json.loads(reply)
        return bool(data.get("contains_almond", False)), int(data.get("count", 0))
    except Exception:
        pass
    contains = "true" in reply.lower() or "yes" in reply.lower()
    nums = re.findall(r"\d+", reply)
    count = int(nums[0]) if nums else (1 if contains else 0)
    return contains, count


def main() -> None:
    if not os.path.isfile(IMAGE_PATH):
        print("false 0")
        return

    raw_b64 = encode_image_b64_raw(IMAGE_PATH)
    reply = call_ollama_with_images(raw_b64, MODEL_NAME)
    contains, count = parse_result(reply)
    print(f"{'true' if contains else 'false'} {count}")


def test_directory(directory_name: str):
    df = pd.read_csv("almond_counts.csv")
    has_almonds_counts = []
    almond_counts = []
    for index, row in df.iterrows():
        filename = row['image_name']
        file_path = os.path.join(directory_name, filename)
        raw_b64 = encode_image_b64_raw(file_path)
        reply = call_ollama_with_images(raw_b64, MODEL_NAME)
        contains, count = parse_result(reply)
        has_almonds_counts.append(contains)
        almond_counts.append(count)
        print(f"{filename}: {'true' if contains else 'false'} {count}")
    df[f"has_almond_{MODEL_NAME}"] = has_almonds_counts
    df[f"almond_count_{MODEL_NAME}"] = almond_counts
    df.to_csv("almond_counts.csv", index=False)


if __name__ == "__main__":
    # main()   # run on single IMAGE_PATH
    test_directory("image_dir")  # batch mode
