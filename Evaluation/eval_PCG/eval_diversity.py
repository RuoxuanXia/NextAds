import os
import re
import json
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash"

REVERSE_PROXY_URL = "path of reverse_proxy"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSy_FAKE_KEY")

MAX_USER_WORKERS = 3

CSV_PATH = "path of results.csv"
RUNS_DIR = "path of ad_runs"

def get_client():
    return genai.Client(
        api_key=GEMINI_API_KEY,
        http_options={
            "base_url": REVERSE_PROXY_URL,
            "api_version": "v1beta"
        }
    )

client = get_client()


def safe_filename(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s).strip())
    return s[:96]

def evaluate_diversity_for_user(uid, video_paths):
    score = 0
    reason = "Analysis failed or error occurred"

    contents = [
        "Evaluate diversity between these ads generated for the SAME USER."
    ]
    
    valid_videos_count = 0

    try:
        for i, path in enumerate(video_paths):
            if not os.path.exists(path):
                continue
                
            try:
                with open(path, "rb") as f:
                    video_bytes = f.read()
                
                video_part = types.Part.from_bytes(
                    data=video_bytes, 
                    mime_type="video/mp4"
                )
                
                contents.append(f"Video #{i+1}:")
                contents.append(video_part)
                valid_videos_count += 1
                
            except Exception as read_err:
                print(f"  [Read Error User {uid}] Could not read {path}: {read_err}")

        if valid_videos_count < 2:
            return uid, 10, "Less than 2 valid videos found, defaulting to max diversity."

        sys_inst = """
    You are an expert at evaluating the diversity of content. Evaluate the diversity between advertisements for the same user on different products, considering the following aspects:
        * **Visual Style**: Is there excessive similarity in tone, lighting, texture, and artistic style across different product ads?
        * **Narrative Form**: Is the narrative structure (tutorial, vlog, plot twist, etc.) and tone (professional, humorous, etc.) overly similar across products?
        * **Subject Matter and Scene Semantics**: Is the subject matter or scene selection overly similar across ads targeting the same user?

    Criteria:
    * **10**: Ads are completely distinct with no noticeable overlap.
    * **7–9**: Ads are mostly distinct with only minor similarities.
    * **4–6**: Ads are somewhat similar but still distinct in key areas.
    * **1–3**: Ads are largely similar with only slight variations.
    * **0**: The ads are identical in all aspects.

    Output JSON strictly: {"score": number, "reason": "brief analysis string"}
    """

        res = client.models.generate_content(
            model=MODEL, 
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=sys_inst, 
                response_mime_type="application/json",
                temperature=0.2
            )
        )
        
        data = json.loads(res.text)
        score = data.get("score", 0)
        reason = data.get("reason", "No reason provided by model.")
        
        return uid, score, reason

    except Exception as e:
        print(f"  [Eval Error User {uid}] {e}")
        return uid, 0, f"Error: {str(e)}"


def main():
    if not os.path.exists(CSV_PATH):
        print("CSV not found.")
        return

    print(f"Reading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    if 'diversity_reason' not in df.columns:
        df['diversity_reason'] = "" 

    pending_users = df[df['diversity_score'] == 0]['user_id'].unique()
    
    print(f"Total Users: {len(df['user_id'].unique())}")
    print(f"Pending Users: {len(pending_users)}")
    
    if len(pending_users) == 0:
        print("All done!")
        return

    user_tasks = []
    for uid in pending_users:
        user_rows = df[df['user_id'] == uid]
        paths = []
        for _, row in user_rows.iterrows():
            safe_prod = safe_filename(row['product_name'])
            path = os.path.join(RUNS_DIR, f"user_{uid}", safe_prod, "video.mp4")
            paths.append(path)
        if paths:
            user_tasks.append((uid, paths))

    print(f"Starting INLINE Mode with Proxy: {REVERSE_PROXY_URL}")

    with ThreadPoolExecutor(max_workers=MAX_USER_WORKERS) as executor:
        future_to_uid = {executor.submit(evaluate_diversity_for_user, uid, paths): uid for uid, paths in user_tasks}
        
        for i, future in enumerate(as_completed(future_to_uid)):
            uid, score, reason = future.result()
            
            df.loc[df['user_id'] == uid, 'diversity_score'] = score
            df.loc[df['user_id'] == uid, 'diversity_reason'] = reason
            
            reason_preview = (reason[:50] + '..') if len(reason) > 50 else reason
            print(f"  [{i+1}/{len(user_tasks)}] User {uid} -> Score: {score} | Reason: {reason_preview}")
            df.to_csv(CSV_PATH, index=False)

    print("Done.")

if __name__ == "__main__":
    main()
