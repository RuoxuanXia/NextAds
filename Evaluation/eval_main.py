import os
import re
import json
import time
import csv
import pickle
import base64
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types


MODEL = "gemini-2.5-flash" 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") 
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL")
MAX_WORKERS = 3
QILIN_ROOT = "path of dataset"
RUNS_DIR = os.path.join(QILIN_ROOT, "ad_runs")
PRODUCT_LIBRARY_JSON = os.path.join(QILIN_ROOT, "product_library", "products_raw.json")
INDEX_DIR = os.path.join(QILIN_ROOT, "index_cache")
USER2RECENT_PKL = os.path.join(INDEX_DIR, "user2recent.pkl")
NOTE_INDEX_PKL = os.path.join(INDEX_DIR, "note_index_v3.pkl") 
OUTPUT_CSV = "/NAS/xiarx/dataset/Qilin/eval_main_new2_results.csv"


def safe_filename(s: str, max_len: int = 96) -> str:
    s = (s or "").strip().replace("’", "_").replace("'", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if len(s) > max_len else s

def get_gemini_client():
    return genai.Client(
        api_key=GEMINI_API_KEY.strip(),
        http_options={
            "base_url": GEMINI_BASE_URL.strip(),
            "api_version": "v1beta"  
        }
    )

client = get_gemini_client()

def get_video_inline_data(video_path: str):
    if not os.path.exists(video_path): return None
    try:
        with open(video_path, "rb") as f:
            encoded_data = base64.b64encode(f.read()).decode('utf-8')
        return types.Part(inline_data=types.Blob(mime_type="video/mp4", data=encoded_data))
    except Exception as e:
        print(f"[Error reading video] {video_path}: {e}")
        return None

def call_gemini_json(contents, sys_inst):
    try:
        res = client.models.generate_content(
            model=MODEL, contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=sys_inst, temperature=0.2, response_mime_type="application/json"
            )
        )
        return json.loads(res.text)
    except Exception as e:
        print(f"  [API Error] {e}")
        return {"score": 0, "reason": str(e)}


def evaluate_presentation(user_note_texts: List[str], video_part: types.Part) -> Dict:
    notes_summary = "\n".join([f"- {t}" for t in user_note_texts[:10]])
    contents = [
        f"User Historical Notes (Style/Vibe):\n{notes_summary}",
        video_part,
        "Task: Evaluate if the video's aesthetic, pacing, and music match the user's vibe."
    ]
    sys_inst = """
    You are an expert in evaluating the presentation of creative content. Given a product and the user's preferences, Evaluate how well the advertisement aligns with the user's aesthetic preferences regarding visual style. Consider the following factors:
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

    Output JSON: {"score": number, "reason": "brief analysis"}
    """
    return call_gemini_json(contents, sys_inst)

def evaluate_content(product_info: Dict, user_note_texts: List[str], video_part: types.Part) -> Dict:
    product_text = f"Product: {product_info.get('product_name')}\nIntro: {product_info.get('product_intro')}"
    notes_summary = "\n".join([f"- {t}" for t in user_note_texts[:10]])
    contents = [
        f"User Interests:\n{notes_summary}",
        product_text,
        video_part,
        "Task: Evaluate if the video highlights product features relevant to THIS user."
    ]
    sys_inst = """
    Evaluate how well the advertisement matches the user’s interest based on the following factors:
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

    Output JSON: {"score": number, "reason": "brief analysis"}
    """
    return call_gemini_json(contents, sys_inst)

def evaluate_alignment(product_info: Dict, video_part: types.Part) -> Dict:
    meta = {"name": product_info.get("product_name"), "slogan": product_info.get("product_slogan")}
    contents = [
        f"Official Metadata: {json.dumps(meta)}",
        video_part,
        "Task: Verify visual identity and factual accuracy."
    ]
    sys_inst = """
    You are an expert at evaluating the consistency of products in images. Given a product and images, Evaluate how well the advertisement matches the user’s interest based on the following factors:
        * **Product Identity Accuracy**: Does the ad represent the product’s appearance, color, model/version, packaging, and accessories correctly?
        * **Brand and Recognition Elements**: Does the ad correctly display the product’s logo, branding, and design language?
        * **Selling Points and Fact Consistency**: Does the ad’s selling point match the product’s real features and details?

    Criteria:
    - **Perfect match (10):** The product in the image matches the actual product in every way.
    - **Good match (7-9):** The product is mostly consistent with minor details changed.
    - **Moderate match (4-6):** The product is partially consistent, but with some significant differences.
    - **Poor match (1-3):** The product in the image is mostly inconsistent with the actual product.
    - **No match (0):** The product identity is completely inconsistent with the actual product.
    Output JSON: {"score": number, "reason": "brief analysis"}
    """
    return call_gemini_json(contents, sys_inst)


def process_single_video(uid, p_name, products_lib, user_texts):
    meta = next((p for p in products_lib if p["product_name"] == p_name), None)
    if not meta:
        meta = next((p for p in products_lib if p["product_name"] == p_name.replace("_", " ")), None)
    if not meta: return None
    safe_name = safe_filename(meta['product_name'])
    video_path = os.path.join(RUNS_DIR, f"user_{uid}", safe_name, "video.mp4")
    video_part = get_video_inline_data(video_path)
    
    if not video_part: return None

    try:
        res_p = evaluate_presentation(user_texts, video_part)
        res_c = evaluate_content(meta, user_texts, video_part)
        res_a = evaluate_alignment(meta, video_part)
    except Exception as e:
        print(f"Error in evaluation for {uid}-{p_name}: {e}")
        return None

    return {
        "user_id": uid,
        "product_name": meta['product_name'],
        
        "presentation_score": res_p.get("score", 0),
        "presentation_reason": res_p.get("reason", "N/A"),
        
        "content_score": res_c.get("score", 0),
        "content_reason": res_c.get("reason", "N/A"),
        
        "identity_consistency_score": res_a.get("score", 0),
        "identity_consistency_reason": res_a.get("reason", "N/A"),
        
        "diversity_score": 0 ,
        "diversity_reason": 0 
    }


def main(input_data):
    print("Loading datasets...")
    with open(PRODUCT_LIBRARY_JSON, "r") as f: products_lib = json.load(f)
    with open(USER2RECENT_PKL, "rb") as f: user2recent = pickle.load(f)
    with open(NOTE_INDEX_PKL, "rb") as f: note_index = pickle.load(f)

    tasks = []
    print(f"Preparing tasks for {len(input_data)} users...")
    
    for uid_str, product_names in input_data.items():
        uid = int(uid_str)
        nids = user2recent.get(uid, [])
        texts = [f"{note_index[int(n)]['title']} {note_index[int(n)]['content']}" 
                 for n in nids[-20:] if int(n) in note_index]

        for p_name in product_names:
            tasks.append((uid, p_name, products_lib, texts))

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
                'user_id', 
                'product_name', 
                'presentation_score', 'presentation_reason',
                'content_score', 'content_reason',
                'identity_consistency_score', 'identity_consistency_reason',
                'diversity_score', 
                'diversity_reason'
            ]
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nDone! Evaluation saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    test_input = {
  "576": [
    "Airbnb",
    "Coca-Cola_Classic",
    "iPhone_17",
    "KFC_Original_Recipe_Chicken",
    "Walmart_Retail"
  ],
  "11094": [
    "Airbnb",
    "L_Or_al_Paris_Revitalift",
    "iPhone_17",
    "Nike_Air_Force_1_07",
    "Coca-Cola_Classic"
  ],
  "502": [
    "iPhone_17",
    "L_Or_al_Paris_Revitalift",
    "McDonald_s_Big_Mac",
    "Nike_Air_Force_1_07",
    "Walmart_Retail"
  ],
  "523": [
    "KFC_Original_Recipe_Chicken",
    "L_Or_al_Paris_Revitalift",
    "McDonald_s_Big_Mac",
    "Coca-Cola_Classic",
    "BMW_3_Series"
  ],
  "1054": [
    "Airbnb",
    "Coca-Cola_Classic",
    "iPhone_17",
    "KFC_Original_Recipe_Chicken",
    "Walmart_Retail"
  ],
  "1453": [
    "Coca-Cola_Classic",
    "iPhone_17",
    "KFC_Original_Recipe_Chicken",
    "McDonald_s_Big_Mac",
    "Walmart_Retail"
  ],
  "5162": [
    "Coca-Cola_Classic",
    "KFC_Original_Recipe_Chicken",
    "Nike_Air_Force_1_07",
    "Subway_Sandwich",
    "Walmart_Retail"
  ],
  "6420": [
    "Airbnb",
    "iPhone_17",
    "Walmart_Retail",
    "BMW_3_Series",
    "Coca-Cola_Classic"
  ],
  "9493": [
    "Airbnb",
    "Coca-Cola_Classic",
    "KFC_Original_Recipe_Chicken",
    "L_Or_al_Paris_Revitalift",
    "Walmart_Retail"
  ],
  "9751": [
    "Airbnb",
    "KFC_Original_Recipe_Chicken",
    "Nike_Air_Force_1_07",
    "Subway_Sandwich",
    "Coca-Cola_Classic"
  ],
  "15067": [
    "Walmart_Retail",
    "McDonald_s_Big_Mac",
    "Coca-Cola_Classic",
    "Airbnb",
    "iPhone_17"
  ],
  "5095": [
    "Coca-Cola_Classic",
    "iPhone_17",
    "KFC_Original_Recipe_Chicken",
    "McDonald_s_Big_Mac",
    "Walmart_Retail"
  ],
  "5282": [
    "Coca-Cola_Classic",
    "Airbnb",
    "iPhone_17",
    "Subway_Sandwich",
    "Walmart_Retail"
  ],
  "4552": [
    "Airbnb",
    "Coca-Cola_Classic",
    "L_Or_al_Paris_Revitalift",
    "iPhone_17",
    "Walmart_Retail"
  ],
  "5302": [
    "Airbnb",
    "Coca-Cola_Classic",
    "KFC_Original_Recipe_Chicken",
    "L_Or_al_Paris_Revitalift",
    "McDonald_s_Big_Mac"
  ],
  "7052": [
    "Coca-Cola_Classic",
    "iPhone_17",
    "KFC_Original_Recipe_Chicken",
    "McDonald_s_Big_Mac",
    "Walmart_Retail"
  ],
  "7469": [
    "Airbnb",
    "KFC_Original_Recipe_Chicken",
    "L_Or_al_Paris_Revitalift",
    "Nike_Air_Force_1_07",
    "Walmart_Retail"
  ],
  "8601": [
    "BMW_3_Series",
    "Coca-Cola_Classic",
    "Lululemon_Align_Pant",
    "Nike_Air_Force_1_07",
    "Walmart_Retail"
  ],
  "11317": [
    "Airbnb",
    "BMW_3_Series",
    "Coca-Cola_Classic",
    "iPhone_17",
    "Walmart_Retail"
  ],
  "12383": [
    "BMW_3_Series",
    "Coca-Cola_Classic",
    "iPhone_17",
    "L_Or_al_Paris_Revitalift",
    "Walmart_Retail"
  ],
  "13731": [
    "Coca-Cola_Classic",
    "KFC_Original_Recipe_Chicken",
    "Lululemon_Align_Pant",
    "Nike_Air_Force_1_07",
    "Walmart_Retail"
  ],
  "6342": [
    "Airbnb",
    "Herman_Miller_Embody_Chair",
    "iPhone_17",
    "KFC_Original_Recipe_Chicken",
    "Walmart_Retail"
  ],
  "6964": [
    "Airbnb",
    "Coca-Cola_Classic",
    "Herman_Miller_Embody_Chair",
    "iPhone_17",
    "Walmart_Retail"
  ],
  "7544": [
    "Airbnb",
    "Coca-Cola_Classic",
    "Herman_Miller_Embody_Chair",
    "iPhone_17",
    "Walmart_Retail"
  ],
  "8150": [
    "Airbnb",
    "Coca-Cola_Classic",
    "iPhone_17",
    "KFC_Original_Recipe_Chicken",
    "Walmart_Retail"
  ],
  "8226": [
    "Airbnb",
    "Coca-Cola_Classic",
    "Herman_Miller_Embody_Chair",
    "KFC_Original_Recipe_Chicken",
    "Walmart_Retail"
  ],
  "8742": [
    "Airbnb",
    "Coca-Cola_Classic",
    "iPhone_17",
    "KFC_Original_Recipe_Chicken",
    "Walmart_Retail"
  ],
  "8931": [
    "Airbnb",
    "Coca-Cola_Classic",
    "iPhone_17",
    "KFC_Original_Recipe_Chicken",
    "Walmart_Retail"
  ],
  "9102": [
    "Airbnb",
    "Coca-Cola_Classic",
    "Herman_Miller_Embody_Chair",
    "iPhone_17",
    "KFC_Original_Recipe_Chicken"
  ],
  "9173": [
    "Airbnb",
    "Coca-Cola_Classic",
    "iPhone_17",
    "KFC_Original_Recipe_Chicken",
    "Walmart_Retail"
  ],
  "9647": [
    "Airbnb",
    "Coca-Cola_Classic",
    "Herman_Miller_Embody_Chair",
    "iPhone_17",
    "KFC_Original_Recipe_Chicken"
  ],
  "10184": [
    "Airbnb",
    "Coca-Cola_Classic",
    "iPhone_17",
    "KFC_Original_Recipe_Chicken",
    "Walmart_Retail"
  ],
  "12687": [
    "Airbnb",
    "Coca-Cola_Classic",
    "iPhone_17",
    "Walmart_Retail",
    "Herman_Miller_Embody_Chair"
  ],
  "12814": [
    "Airbnb",
    "Colgate_Total_Toothpaste",
    "Herman_Miller_Embody_Chair",
    "iPhone_17",
    "Walmart_Retail"
  ],
  "12907": [
    "Airbnb",
    "Coca-Cola_Classic",
    "Herman_Miller_Embody_Chair",
    "iPhone_17",
    "KFC_Original_Recipe_Chicken"
  ],
  "13386": [
    "Airbnb",
    "Coca-Cola_Classic",
    "Herman_Miller_Embody_Chair",
    "iPhone_17",
    "KFC_Original_Recipe_Chicken"
  ],
  "14037": [
    "Airbnb",
    "Coca-Cola_Classic",
    "Herman_Miller_Embody_Chair",
    "KFC_Original_Recipe_Chicken",
    "Walmart_Retail"
  ],
  "14139": [
    "Colgate_Total_Toothpaste",
    "Herman_Miller_Embody_Chair",
    "iPhone_17",
    "KFC_Original_Recipe_Chicken",
    "Walmart_Retail"
  ],
  "14150": [
    "Airbnb",
    "Coca-Cola_Classic",
    "Herman_Miller_Embody_Chair",
    "iPhone_17",
    "KFC_Original_Recipe_Chicken"
  ],
  "14456": [
    "Airbnb",
    "Coca-Cola_Classic",
    "KFC_Original_Recipe_Chicken",
    "Walmart_Retail",
    "Herman_Miller_Embody_Chair"
  ],
  "14460": [
    "Airbnb",
    "Coca-Cola_Classic",
    "Herman_Miller_Embody_Chair",
    "Snow_Peak_Amenity_Dome_Tent",
    "Walmart_Retail"
  ],
  "14739": [
    "Airbnb",
    "Coca-Cola_Classic",
    "Herman_Miller_Embody_Chair",
    "KFC_Original_Recipe_Chicken",
    "Walmart_Retail"
  ],
  "15256": [
    "Airbnb",
    "Coca-Cola_Classic",
    "Herman_Miller_Embody_Chair",
    "KFC_Original_Recipe_Chicken",
    "Walmart_Retail"
  ],
  "6747": [
    "Airbnb",
    "Coca-Cola_Classic",
    "iPhone_17",
    "Snow_Peak_Amenity_Dome_Tent",
    "Walmart_Retail"
  ],
  "6756": [
    "Airbnb",
    "Coca-Cola_Classic",
    "iPhone_17",
    "Snow_Peak_Amenity_Dome_Tent",
    "Walmart_Retail"
  ],
  "10538": [
    "Airbnb",
    "Colgate_Total_Toothpaste",
    "iPhone_17",
    "Walmart_Retail",
    "Coca-Cola_Classic"
  ],
  "9385": [
    "Airbnb",
    "Colgate_Total_Toothpaste",
    "iPhone_17",
    "KFC_Original_Recipe_Chicken",
    "Walmart_Retail"
  ],
  "2198": [
    "Coca-Cola_Classic",
    "Colgate_Total_Toothpaste",
    "iPhone_17",
    "KFC_Original_Recipe_Chicken",
    "Walmart_Retail"
  ],
  "6025": [
    "Airbnb",
    "Coca-Cola_Classic",
    "Colgate_Total_Toothpaste",
    "iPhone_17",
    "Snow_Peak_Amenity_Dome_Tent"
  ],
  "6207": [
    "Airbnb",
    "Coca-Cola_Classic",
    "Colgate_Total_Toothpaste",
    "KFC_Original_Recipe_Chicken",
    "Walmart_Retail"
  ]
}
    main(test_input)