import os
import re
import json
import time
import base64
import uuid
import unicodedata
from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple

import requests
import httpx
from PIL import Image, UnidentifiedImageError
from openai import OpenAI
import oss2

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
TEMP_TIER = 0.2
TEMP_MATCH = 0.2
TEMP_PICK = 0.2
TEMP_VTOK = 0.2
TEMP_STORY = 0.6

MAX_TEXT_CHARS = 700
GRID_TILE = 384
SORA_REF_TARGET_H = 768

OSS_ACCESS_KEY_ID = os.environ.get("OSS_ACCESS_KEY_ID")
OSS_ACCESS_KEY_SECRET = os.environ.get("OSS_ACCESS_KEY_SECRET")
OSS_ENDPOINT = "oss-ap-southeast-1.aliyuncs.com" 
OSS_BUCKET_NAME = "sora-ref-images"

HTTP_TIMEOUT = 60
OPENAI_TIMEOUT_SEC = 75

SORA2_MODEL = "sora-2-pro"
SORA2_DURATION = 15
SORA2_ASPECT = "9:16"
SORA2_WATERMARK = False

BANNED_TOKENS = {
    "hamster", "仓鼠",
    "jk", "JK",
    "lolita", "Lolita", "洛丽塔", "lo裙",
    "vtuber", "二次元", "coser", "cosplay"
}

WORKSPACE_DIR = "."
HISTORY_DIR = os.path.join(WORKSPACE_DIR, "history")
PRODUCT_DIR = os.path.join(WORKSPACE_DIR, "product")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_text(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKC", s)
    return " ".join(s.split())

def truncate_text(s: str, n: int) -> str:
    s = normalize_text(s)
    return s if len(s) <= n else s[:n] + "…"

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: Any):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _must_be_ascii(name: str, v: str):
    try:
        (v or "").encode("ascii")
    except Exception:
        raise RuntimeError(f"{name} must be ASCII.")

def open_image_rgb(path: str) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except UnidentifiedImageError:
        with open(path, "rb") as f:
            b = f.read()
        return Image.open(BytesIO(b)).convert("RGB")

def _resize_keep_ratio(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    scale = max_side / max(w, h)
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

def parse_llm_json_output(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            cand = parts[1]
            cand = cand.lstrip("json").strip()
            try:
                return json.loads(cand)
            except Exception:
                pass
    try:
        return json.loads(text)
    except Exception:
        return {"_raw": text, "_error": "PARSE_ERROR"}

def _sanitize_banned_tokens(obj: Any) -> Any:
    if isinstance(obj, str):
        s = obj
        for t in BANNED_TOKENS:
            s = re.sub(rf"\b{re.escape(t)}\b", "", s, flags=re.IGNORECASE)
            s = s.replace(t, "")
        return " ".join(s.split())
    if isinstance(obj, list):
        return [_sanitize_banned_tokens(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _sanitize_banned_tokens(v) for k, v in obj.items()}
    return obj

def make_2x2_grid(image_paths: List[str], tile_size: int = GRID_TILE, bg=(255, 255, 255)) -> Optional[Image.Image]:
    valid = [p for p in image_paths if p and os.path.exists(p)][:4]
    if not valid:
        return None
    canvas = Image.new("RGB", (tile_size * 2, tile_size * 2), bg)
    for i in range(4):
        x0, y0 = (i % 2) * tile_size, (i // 2) * tile_size
        if i < len(valid):
            try:
                img = _resize_keep_ratio(open_image_rgb(valid[i]), tile_size)
                canvas.paste(img, (x0 + (tile_size - img.width) // 2, y0 + (tile_size - img.height) // 2))
            except Exception:
                pass
    return canvas

def concat_left_right_keep_height(left: Image.Image, right: Image.Image, target_h: int = SORA_REF_TARGET_H, gap: int = 32) -> Image.Image:
    def resize_to_h(img: Image.Image, h: int) -> Image.Image:
        w0, h0 = img.size
        s = h / h0
        return img.resize((max(1, int(w0 * s)), h), Image.LANCZOS)
    L = resize_to_h(left, target_h)
    R = resize_to_h(right, target_h)
    W = L.width + gap + R.width
    canvas = Image.new("RGB", (W, target_h), (255, 255, 255))
    canvas.paste(L, (0, 0))
    canvas.paste(R, (L.width + gap, 0))
    return canvas

def get_openai_client() -> OpenAI:
    api_key = (os.environ.get("OPENAI_API_KEY", "") or "").strip()
    base_url = (os.environ.get("OPENAI_BASE_URL", "") or "").strip()
    _must_be_ascii("OPENAI_API_KEY", api_key)
    http_client = httpx.Client(timeout=httpx.Timeout(OPENAI_TIMEOUT_SEC))
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
    return OpenAI(api_key=api_key, http_client=http_client)

client = get_openai_client()

def llm_chat_json(messages: List[Dict[str, Any]], temperature: float, max_retries: int = 3) -> Dict[str, Any]:
    last_err = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=temperature)
            return parse_llm_json_output(resp.choices[0].message.content)
        except Exception as e:
            last_err = e
            time.sleep(4)
    raise RuntimeError(f"OpenAI call failed after {max_retries} tries: {last_err}")

def encode_images_for_vision(paths: List[str], max_size=(512, 512), quality=80) -> List[Dict[str, Any]]:
    items = []
    for p in paths:
        try:
            img = open_image_rgb(p)
            img.thumbnail(max_size)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            items.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        except Exception:
            continue
    return items

def prompt_text_tiering_v3() -> str:
    return normalize_text("""
You are a user preference clustering engine.
Input:
- A user's recently clicked notes. Each note has:
  note_idx, title, content (truncated)
Goal:
Create THREE tiers of preference strength based on frequency and consistency of ONE dominant preference per tier.
Hard constraints (very important):
1) Each tier must represent EXACTLY ONE preference theme (one coherent interest).
   - No mixing multiple unrelated interests in the same tier.
2) Use frequency/coverage to decide strength:
   - strong: the most frequent & consistent dominant theme
   - medium: the second most frequent coherent theme (still meaningful)
   - weak: the remaining coherent theme OR "misc/rare"
3) If you cannot find 3 coherent themes, you must still output 3 tiers:
   - weak can be "misc/rare" with minimal note_idxs.
4) DO NOT introduce new entities that are not explicitly present in TITLES.
5) Output should be concise conclusions only. No reasoning steps.
Also output global text style:
- tone_tags: a few tags
- info_density: low | medium | high
Output JSON ONLY:
{
  "tiers": {
    "strong": {"theme": "...", "summary": "...<=40 words", "note_idxs": [...]},
    "medium": {"theme": "...", "summary": "...<=40 words", "note_idxs": [...]},
    "weak": {"theme": "misc/rare|...", "summary": "...<=30 words", "note_idxs": [...]}
  },
  "text_style_global": {"tone_tags": [...], "info_density": "low|medium|high"}
}
""").strip()

def prompt_choose_tier_v3() -> str:
    return normalize_text("""
You are an expert in personalized in-feed native video ads.
We will choose ONLY between tier "strong" and tier "medium".
Tier "weak" must NEVER be selected.
Task:
Given user tier summaries and product info, score BOTH strong and medium for:
A) **product_match (0-10)**: How well does this user's preference align with the product? A higher score means the product is highly relevant to the user’s interests, creating a more personalized ad.
B) **ad_ability (0-10)**: How easy is it to create a believable native ad from this tier? Does the ad feel authentic and seamless, fitting naturally within the user's content?
C) **expressibility (0-10)**: How easy is it to craft a 15-second ad with a clear structure (hook -> value -> proof -> CTA) using this tier's preference?
Then, choose the tier that has the **best potential** to generate a **personalized, effective ad** based on the user's preferences and how well they align with the product.
- If the **total score difference** between "strong" and "medium" for **product match**, **ad ability**, and **expressibility** is **greater than 15 points**, select the tier with the **higher total score**.
- If the **total score difference** is **less than or equal to 15 points**, choose the tier that has the **better personalization potential** for the user, meaning the one that will create the most engaging, relevant ad, with a focus on personalization.
Constraints:
- Use **ONLY** provided tier summaries/themes and product text.
- Do **NOT** invent external facts or new entities.
- Provide a concise reason (<=90 words) explaining why the selected tier is the most **effective for personalized ad generation**, focusing on how well it matches the user’s preferences and generates an engaging ad.
Output JSON ONLY:
{
  "candidates": {
    "strong": {"product_match": 0, "ad_ability": 0, "expressibility": 0, "total": 0, "reason": "..."},
    "medium": {"product_match": 0, "ad_ability": 0, "expressibility": 0, "total": 0, "reason": "..."}
  },
  "best_tier": "strong|medium",
  "confidence": 0.0,
  "suggested_hooks": ["...", "..."]
}
""").strip()

def prompt_pick_rep_images_v3(level: str) -> str:
    return normalize_text(f"""
You are a visual preference selector.
Input: images from the SAME user and SAME tier = "{level}".
Task:
Select up to 4 images that best represent the user's recurring visual preference
AND are "reusable" for a generic native ad vibe.
Rules:
- Prefer images with clear scene/style and safe reusable elements (lighting, environment, props).
- Avoid overly specific/rare story elements.
- Output JSON only with chosen indices in input order.
Schema:
{{
  "level": "{level}",
  "chosen": [0, 1, 2, 3],
  "notes": "one short sentence"
}}
""").strip()

def prompt_visual_elements_v3() -> str:
    return normalize_text("""
You extract BOTH:
(1) vibe/style summary
(2) concrete reusable visual elements
Input: a 2x2 grid of representative user images.
Output JSON ONLY:
{
  "vibe": {
    "mood_tags": ["..."],
    "colors_lighting": ["..."],
    "camera_style": ["..."]
  },
  "elements": {
    "scenes": ["generic scene types"],
    "objects": ["generic props/objects"],
    "actions": ["generic actions/motions"],
    "composition": ["composition cues"]
  }
}
Hard constraints:
- Do NOT name specific people, celebrities, brands (unless clearly visible), or unique identities.
- Do NOT output niche subculture/entity tokens (e.g., hamster / JK / Lolita).
- Keep everything generic and reusable in ads.
""").strip()

def prompt_storyboard_v3() -> str:
    return normalize_text("""
You are a vertical short-form ad director and copywriter.
Create a 15-second 9:16 in-feed native ad storyboard.
The ad must feel like organic content and must NOT be static.
Inputs:
- user_demographics (age/gender or unknown)
- text_style_global (tone_tags, info_density)
- tier_used (theme, summary) + tier_notes (titles + truncated content)
- visual (vibe + concrete elements) extracted from the user's images
- product info (name, slogan, intro, details)
- suggested_hooks
Hard constraints:
1) Ensure the text is spelled correctly with 100% accuracy.
2) Must change at least every 2-3 seconds (new shot, new action, new angle).
3) Reserve the LAST 2 seconds for CTA and a clear product hero shot.
4) Voiceover/subtitles must fit 15s:
   - keep each scene's voiceover <= 16 English words
   - keep each subtitle <= 8 words
5) Do NOT invent specific animals/characters or niche entities.
6) Visual personalization must be grounded in provided visual elements (scenes/objects/actions/colors).
7) Product depiction must be faithful; no wrong logos; no shape distortion.
Output ENGLISH JSON ONLY:
{
  "duration_sec": 15,
  "format": "9:16",
  "tier_used": "strong|medium",
  "target_user": {"age": "...", "gender": "..."},
  "tone_tags": ["..."],
  "info_density": "low|medium|high",
  "hook": "one sentence hook",
  "scenes": [
    {
      "sec": "0-2",
      "shot_goal": "what this shot accomplishes",
      "visual": "what we see (use visual elements)",
      "camera_motion": "push-in/pan/handheld walk/etc",
      "action": "what moves/changes",
      "subtitle": "short on-screen text",
      "voiceover": "short voiceover",
      "product_focus": "feature or benefit",
      "personalization_used": ["..."]
    }
  ],
  "cta": "call to action line",
  "timing_notes": {
    "last_2s_reserved_for_cta": true,
    "anti_static_rule": "Every scene has motion + change"
  }
}
""").strip()

def prompt_sora_header_v3() -> str:
    return normalize_text("""
Generate a 15-second vertical (9:16) in-feed native video ad.
Reference image usage:
- The LEFT part (2x2 grid) provides the audience's visual vibe and recurring elements.
- The RIGHT part is the official product image that must be reproduced faithfully
  (no shape distortion, no wrong logos).
Creative constraints:
- Keep it organic, modern, realistic (like social content).
- Use the LEFT-side vibe as inspiration, but do not copy any specific person identity.
- Reuse ONLY generic visual elements visible in LEFT images.
- Product depiction must match RIGHT image accurately.
- Must not be static; ensure continuous motion and shot changes.
- Avoid freeze-frame or long still shots.
- Keep last ~2 seconds for a clear product hero shot + CTA.
Now generate the video based on the storyboard JSON below.
""").strip()

def upload_to_aliyun_oss(local_file_path: str) -> Optional[str]:
    if not local_file_path or not os.path.exists(local_file_path):
        return None
    if not OSS_ACCESS_KEY_ID or not OSS_ACCESS_KEY_SECRET:
        print("  [OSS Error] Missing Aliyun Access Keys.")
        return None
    try:
        auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
        bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET_NAME)
        ext = os.path.splitext(local_file_path)[-1]
        remote_name = f"sora_ref/demo_{uuid.uuid4()}{ext}"
        bucket.put_object_from_file(remote_name, local_file_path)
        return f"https://{OSS_BUCKET_NAME}.{OSS_ENDPOINT}/{remote_name}"
    except Exception as e:
        print(f"  [OSS Error] {e}")
        return None

def sora2_submit(prompt_text: str, ref_image_path: Optional[str]) -> str:
    url = "https://api.apimart.ai/v1/videos/generations"
    tok = os.environ.get("APIMART_TOKEN", "").strip()
    headers = {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}
    payload = {
        "model": SORA2_MODEL, "prompt": prompt_text,
        "duration": SORA2_DURATION, "aspect_ratio": SORA2_ASPECT, "watermark": SORA2_WATERMARK
    }
    if ref_image_path and os.path.exists(ref_image_path):
        public_url = upload_to_aliyun_oss(ref_image_path)
        if public_url:
            payload["image_urls"] = [public_url]
    r = requests.post(url, json=payload, headers=headers, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()["data"][0]["task_id"]


def run_local_demo():
    print("NextAds_Demo starts")
    
    # [0] Load Local Folder Data
    notes_payload = read_json(os.path.join(HISTORY_DIR, "notes.json"))
    for note in notes_payload:
        note["content"] = truncate_text(note.get("content", ""), MAX_TEXT_CHARS)
        
    product_info = read_json(os.path.join(PRODUCT_DIR, "info.json"))
    product_img_path = os.path.join(PRODUCT_DIR, "product_image.jpg")
    history_image_paths = [os.path.join(HISTORY_DIR, f) for f in os.listdir(HISTORY_DIR) if f.lower().endswith(('.jpg', '.png'))]

    # [1] Text Tiering
    print("  [1/8] get_text_tiering...")
    prompt = prompt_text_tiering_v3()
    input_json = {"user_id": "demo_user", "notes": notes_payload}
    tier_out = _sanitize_banned_tokens(llm_chat_json([{"role": "user", "content": prompt + "\n\nINPUT_JSON:\n" + json.dumps(input_json, ensure_ascii=False)}], temperature=TEMP_TIER))
    write_json(os.path.join(OUTPUT_DIR, "tiering.json"), tier_out)

    # [2] Match Best Tier
    print("  [2/8] choose_best_tier_for_product...")
    prompt_match = prompt_choose_tier_v3()
    tiers = tier_out.get("tiers", {})
    user_tiers_payload = {
        "strong": {"theme": tiers.get("strong", {}).get("theme", ""), "summary": tiers.get("strong", {}).get("summary", "")},
        "medium": {"theme": tiers.get("medium", {}).get("theme", ""), "summary": tiers.get("medium", {}).get("summary", "")}
    }
    match_input = {
        "user_tiers": user_tiers_payload,
        "text_style_global": tier_out.get("text_style_global", {}),
        "product": product_info
    }
    match_result = _sanitize_banned_tokens(llm_chat_json([{"role": "user", "content": prompt_match + "\n\nINPUT_JSON:\n" + json.dumps(match_input, ensure_ascii=False)}], temperature=TEMP_MATCH))
    best_tier = match_result.get("best_tier", "medium")
    write_json(os.path.join(OUTPUT_DIR, "match_result.json"), match_result)

    # [3] Pick Images (Vision LLM)
    print(f"  [3/8] {best_tier} pick_representative_images...")
    if len(history_image_paths) > 4:
        vision_items = encode_images_for_vision(history_image_paths, max_size=(512, 512), quality=80)
        pick_prompt = prompt_pick_rep_images_v3(best_tier)
        pick_out = _sanitize_banned_tokens(llm_chat_json([{"role": "user", "content": [{"type": "text", "text": pick_prompt}, *vision_items]}], temperature=TEMP_PICK))
        chosen_idx = pick_out.get("chosen", [0,1,2,3])
        chosen_paths = [history_image_paths[i] for i in chosen_idx if i < len(history_image_paths)][:4]
    else:
        chosen_paths = history_image_paths

    # [4] Build Reference Images
    print("  [4/8] Constructing Visual Reference Grids and Stitching...")
    user_grid = make_2x2_grid(chosen_paths, tile_size=GRID_TILE)
    if user_grid:
        user_grid_path = os.path.join(OUTPUT_DIR, "user_grid.jpg")
        user_grid.save(user_grid_path, "JPEG", quality=92)
        if os.path.exists(product_img_path):
            prod_img = open_image_rgb(product_img_path)
            sora_ref = concat_left_right_keep_height(user_grid, prod_img, target_h=SORA_REF_TARGET_H, gap=32)
            sora_ref_path = os.path.join(OUTPUT_DIR, "sora_ref.jpg")
            sora_ref.save(sora_ref_path, "JPEG", quality=92)
        else:
            sora_ref_path = user_grid_path
    else:
        user_grid_path, sora_ref_path = None, None

    # [5] Extract Visual Elements
    print("  [5/8] extract_visual_elements...")
    if user_grid_path:
        v_items = encode_images_for_vision([user_grid_path], max_size=(768, 768), quality=85)
        visual = _sanitize_banned_tokens(llm_chat_json([{"role": "user", "content": [{"type": "text", "text": prompt_visual_elements_v3()}, *v_items]}], temperature=TEMP_VTOK))
    else:
        visual = {"vibe": {}, "elements": {}}
    write_json(os.path.join(OUTPUT_DIR, "visual_elements.json"), visual)

    # [6] Storyboard
    print("  [6/8] generate_storyboard...")
    story_input = {
        "user_demographics": {"age": "unknown", "gender": "unknown"},
        "text_style_global": tier_out.get("text_style_global", {}),
        "tier_used": {"tier": best_tier, "theme": tiers.get(best_tier, {}).get("theme", ""), "summary": tiers.get(best_tier, {}).get("summary", ""), "tier_notes": notes_payload},
        "visual": visual,
        "product": product_info,
        "suggested_hooks": match_result.get("suggested_hooks", [])
    }
    storyboard = _sanitize_banned_tokens(llm_chat_json([{"role": "user", "content": prompt_storyboard_v3() + "\n\nINPUT_JSON:\n" + json.dumps(story_input, ensure_ascii=False)}], temperature=TEMP_STORY))
    write_json(os.path.join(OUTPUT_DIR, "storyboard.json"), storyboard)

    # [7] Sora Web Prompt
    print("  [7/8] Assemble Sora submission text...")
    sora_web_prompt = prompt_sora_header_v3() + "\n\nSTORYBOARD_JSON:\n" + json.dumps(storyboard, ensure_ascii=False, indent=2) + "\n"
    with open(os.path.join(OUTPUT_DIR, "sora_web_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(sora_web_prompt)

    # [8] Submit Task
    print("  [8/8] submit to the video generation API...")
    try:
        task_id = sora2_submit(sora_web_prompt, sora_ref_path)
        write_json(os.path.join(OUTPUT_DIR, "sora_task.json"), {"task_id": task_id, "status": "submitted"})
        print(f"✅ success！Task ID: {task_id} (saved in {OUTPUT_DIR}/)")
    except Exception as e:
        print(f"❌ fail: {e}")

if __name__ == "__main__":
    run_local_demo()
