import os
import re
import json
import time
import csv
import numpy as np
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types

# ================= Configurations =================
MODEL = "gemini-2.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") 
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL")

MAX_WORKERS = 16
DATA_ROOT = "path/to/your/data"
VIDEO_DIR = os.path.join(DATA_ROOT, "gen_ads")
COVER_DIR = os.path.join(DATA_ROOT, "covers")
PRODUCT_JSON = os.path.join(DATA_ROOT, "products.json")
USER_JSON = os.path.join(DATA_ROOT, "users.json")
OUTPUT_CSV = "evaluation_results.csv"

# ================= Tools =================
def get_gemini_client():
    return genai.Client(
        api_key=GEMINI_API_KEY.strip(),
        http_options={
            "base_url": GEMINI_BASE_URL.strip(),
            "api_version": "v1beta"
        }
    )

client = get_gemini_client()

def get_media_part(path: str, mime_type: str):
    if not os.path.exists(path): return None
    try:
        with open(path, "rb") as f:
            data = f.read()
        return types.Part.from_bytes(data=data, mime_type=mime_type)
    except Exception as e:
        print(f"[Error reading file] {path}: {e}")
        return None

def call_gemini_json(contents: List[Any], sys_inst: str):
    for i in range(3):
        try:
            res = client.models.generate_content(
                model=MODEL, 
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=sys_inst, 
                    temperature=0.2, 
                    response_mime_type="application/json"
                )
            )
            return json.loads(res.text)
        except Exception as e:
            if i == 2: return {"score": 0, "reason": f"API Error: {str(e)}"}
            time.sleep(2)

# ================= Evaluation dimensions =================
def evaluate_presentation(image_parts: List[types.Part], video_part: types.Part) -> Dict:
    sys_inst = """You are an expert in evaluating the presentation of creative content. Given a product and the covers of the user's interacted videos, Evaluate how well the advertisement aligns with the user's aesthetic preferences regarding visual style. Consider the following factors:

* **Tone**: Does the tone match the user’s preference (warm, cold, neutral)?
* **Lighting**: Is the lighting appropriate for the user’s preference (soft, harsh, natural)?
* **Texture**: Does the texture (smooth, rough, etc.) fit with the user’s aesthetic?
* **Artistic Style**: Does the artistic style (minimalistic, modern, vintage, etc.) align with the user’s taste?

Criteria:
- **Perfect match (10):** The content presentation matches perfectly with the user's preferences.
- **Good match (7-9):** The content presentation aligns well but has some minor differences.
- **Moderate match (4-6):** The content presentation partially matches the user's preferences.
- **Poor match (1-3):** The content presentation does not align well with the user's preferences.
- **No match (0):** The content presentation completely fails to align with the user's preferences.

Output JSON: 
```json
{   
    "score": number, 
    "reason": "brief analysis"
}
```"""
    contents = image_parts + [video_part, "Task: Evaluate aesthetic alignment."]
    return call_gemini_json(contents, sys_inst)

def evaluate_content(image_parts: List[types.Part], video_part: types.Part) -> Dict:
    sys_inst = """You are an expert in evaluating how well the advertisement matches the user’s interest. Given a product and the covers of the user's interacted videos, Evaluate how well the advertisement aligns with the user's aesthetic preferences regarding visual style. Consider the following factors:

* **Subject Matter**: Does the subject matter of the advertisement align with the user’s preferences?
* **Scene Semantics**: Does the scene selection (location, setting) align with what the user would enjoy?
* **Information Angle**: Does the advertisement focus on the aspects the user cares about (performance, aesthetics, value for money)?
* **Appeal**: Does the ad focus on motivations that resonate with the user (saving money, efficiency, quality, aesthetics, health, eco-friendliness)?
    
Criteria:
- **Perfect match (10):** The content is perfectly aligned with the user's preferences.
- **Good match (7-9):** The content aligns well but with some differences.
- **Moderate match (4-6):** The content partially aligns with the user's preferences.
- **Poor match (1-3):** The content does not align well with the user's preferences.
- **No match (0):** The content is completely irrelevant to the user.

Output JSON: 
```json
{   
    "score": number, 
    "reason": "brief analysis"
}
```"""
    contents = image_parts + [video_part, "Task: Evaluate content interest."]
    return call_gemini_json(contents, sys_inst)

def evaluate_alignment(product_image_part: types.Part, video_part: types.Part) -> Dict:
    sys_inst = """You are an expert at evaluating the consistency of products in the advertisement. Given a product and the video, Evaluate how well the advertisement matches the product based on the following factors:

* **Product Identity Accuracy**: Does the ad represent the product’s appearance, color, model/version, packaging, and accessories correctly?
* **Brand and Recognition Elements**: Does the ad correctly display the product’s logo, branding, and design language?
* **Selling Points and Fact Consistency**: Does the ad’s selling point match the product’s real features and details?

Criteria:
- **Perfect match (10):** The product in the video matches the actual product in every way.
- **Good match (7-9):** The product is mostly consistent with minor details changed.
- **Moderate match (4-6):** The product is partially consistent, but with some significant differences.
- **Poor match (1-3):** The product in the video is mostly inconsistent with the actual product.
- **No match (0):** The product identity is completely inconsistent with the actual product.

Output JSON: 
```json
{   
    "score": number, 
    "reason": "brief analysis"
}
```"""
    contents = [product_image_part, video_part, "Task: Verify product identity."]
    return call_gemini_json(contents, sys_inst)

def evaluate_integration(image_parts: List[types.Part], video_part: types.Part) -> Dict:
    sys_inst = """You are an expert in evaluating the seamlessness and integration of advertisement within video content. Given a video containing an advertisement, evaluate how naturally the ad is integrated or how disruptive (intrusive) it feels to the viewer's experience. Consider the following factors:

* **Contextual Flow**: Does the transition between the organic content and the advertisement feel smooth, or is it a sudden, jarring break?
* **Thematic Consistency**: Does the ad share a similar tone, visual style, or topic with the surrounding video content?
* **User Experience Disruption**: Does the ad interrupt a high-stakes moment (e.g., a climax or a key explanation), or is it placed at a logical breaking point?
* **Narrative Integration**: Is the ad woven into the story/dialogue (native advertising), or is it an external overlay/hard cut that pulls the viewer out of the experience?

Criteria:

* **Seamless (10):** The ad is perfectly integrated; the viewer might not even perceive it as a disruption.
* **Natural (7-9):** The ad is well-placed and shares thematic elements with the video, causing minimal friction.
* **Noticeable but Acceptable (4-6):** The ad is clearly a break from content, but occurs at a logical pause or maintains a similar aesthetic.
* **Intrusive (1-3):** The ad feels forced, breaks the immersion significantly, or uses a jarringly different tone/volume.
* **Highly Disruptive (0):** The ad completely ruins the viewing experience through poor timing, extreme contrast, or aggressive interruption.

Output JSON:
```json
{
    "score": number, 
    "reason": "brief analysis of the transition and contextual fit"
}
```"""
    contents = image_parts + [video_part, "Task: Evaluate integration alignment."]
    return call_gemini_json(contents, sys_inst)

def process_single_video(uid: int, pid: str, user_data: Dict, product_data: Dict):
    interacted_vids = user_data['vids'][:-1]
    image_parts = [get_media_part(os.path.join(COVER_DIR, f"{vid}.jpg"), "image/jpeg") for vid in interacted_vids]
    image_parts = [p for p in image_parts if p]
    
    video_path = os.path.join(VIDEO_DIR, f"user_{uid}_product_{pid}.mp4")
    video_part = get_media_part(video_path, "video/mp4")
    
    product_image_part = get_media_part(product_data.get('product_image', ''), "image/jpeg")

    if not video_part: return None

    res_p = evaluate_presentation(image_parts, video_part)
    res_c = evaluate_content(image_parts, video_part)
    res_a = evaluate_alignment(product_image_part, video_part) if product_image_part else {"score":0, "reason":"No product img"}
    res_i = evaluate_integration(image_parts, video_part)

    return {
        "user_id": uid,
        "product_id": pid,
        "product_name": product_data.get('product_name', 'N/A'),
        
        "presentation_score": res_p.get("score", 0),
        "presentation_reason": res_p.get("reason", "N/A"),
        
        "content_score": res_c.get("score", 0),
        "content_reason": res_c.get("reason", "N/A"),
        
        "identity_consistency_score": res_a.get("score", 0),
        "identity_consistency_reason": res_a.get("reason", "N/A"), 

        "integration_quality_score": res_i.get("score", 0), 
        "integration_quality_reason": res_i.get("reason", "N/A")
    }

def main():
    print("Loading datasets...")
    with open(PRODUCT_JSON, "r") as f: all_products = json.load(f)
    with open(USER_JSON, "r") as f: all_users = json.load(f)

    tasks = []
    for uid_idx, user in enumerate(all_users):
        for pid in user.get('products', []):
            if pid in all_products:
                tasks.append((uid_idx, pid, user, all_products[pid]))

    results = []
    print(f"Starting execution with {MAX_WORKERS} threads for {len(tasks)} videos...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(process_single_video, *t): t for t in tasks}
        
        for i, future in enumerate(as_completed(future_to_task)):
            res = future.result()
            if res:
                results.append(res)
            
            if (i + 1) % 5 == 0:
                print(f"  Progress: {i + 1}/{len(tasks)} completed.")

    if results:
        with open(OUTPUT_CSV, "w", newline='', encoding='utf-8') as f:
            headers = [
                'user_id', 'product_id', 'product_name', 
                'presentation_score', 'presentation_reason',
                'content_score', 'content_reason',
                'identity_consistency_score', 'identity_consistency_reason', 
                'integration_quality_score', 'integration_quality_reason'
            ]
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)
        
        avg_score = np.mean([r['presentation_score'] for r in results])
        print(f"\nDone! Evaluation saved to {OUTPUT_CSV}. Avg Presentation: {avg_score:.2f}")

if __name__ == "__main__":
    main()
