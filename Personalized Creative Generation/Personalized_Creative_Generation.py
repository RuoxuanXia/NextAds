import os
import re
import json
import time
import base64
import pickle
import argparse
import unicodedata
from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import httpx
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image, UnidentifiedImageError
from openai import OpenAI
import oss2  
import uuid  


# -----------------------------
# Global config
# -----------------------------
QILIN_ROOT = "path of qilin dataset"

NOTES_GLOB = os.path.join(QILIN_ROOT, "notes", "*.parquet")
REC_TRAIN_GLOB = os.path.join(QILIN_ROOT, "recommendation_train", "*.parquet")
USER_FEAT_GLOB = os.path.join(QILIN_ROOT, "user_feat", "*.parquet")

IMAGES_ROOT = 'path of pictures of qilin dataset'

PRODUCT_LIBRARY_JSON = os.path.join(QILIN_ROOT, "product_library", "products.json")

INDEX_DIR = os.path.join(QILIN_ROOT, "index_cache")
os.makedirs(INDEX_DIR, exist_ok=True)

USER2RECENT_PKL = os.path.join(INDEX_DIR, "user2recent.pkl")
USER_FEAT_PKL = os.path.join(INDEX_DIR, "user_feat.pkl")

NOTE_INDEX_PKL_V3 = os.path.join(INDEX_DIR, "note_index.pkl")
MAX_IMAGES_PER_NOTE_INDEX = 8  

USER_PROFILE_DIR = os.path.join(QILIN_ROOT, "user_profiles")
os.makedirs(USER_PROFILE_DIR, exist_ok=True)

RUNS_DIR = os.path.join(QILIN_ROOT, "ad_runs")
os.makedirs(RUNS_DIR, exist_ok=True)

# LLM
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
TEMP_TIER = 0.2
TEMP_MATCH = 0.2
TEMP_PICK = 0.2
TEMP_VTOK = 0.2
TEMP_STORY = 0.6

# limits
MAX_NOTES_FOR_TIERING = 60
MAX_TEXT_CHARS = 700            
MAX_NOTES_FOR_SCRIPT = 18       
MAX_CANDIDATE_IMAGES_TOTAL = 60
MAX_IMAGES_PER_NOTE_FOR_CANDIDATES = 4

# image build
GRID_TILE = 384
SORA_REF_TARGET_H = 768
OSS_ACCESS_KEY_ID = os.environ.get("OSS_ACCESS_KEY_ID")
OSS_ACCESS_KEY_SECRET = os.environ.get("OSS_ACCESS_KEY_SECRET")
OSS_ENDPOINT = "oss_endpoint" 
OSS_BUCKET_NAME = "oss_bucket_name"

# network
HTTP_TIMEOUT = 60
OPENAI_TIMEOUT_SEC = 75

# sora2
SORA2_MODEL = "sora-2"
SORA2_DURATION = 15
SORA2_ASPECT = "9:16"
SORA2_WATERMARK = False

POLL_BASE_INTERVAL = 10
POLL_MAX_SEC = 3600  # 60min
MAX_POLL_ERRORS_BEFORE_RESUBMIT = 8
MAX_RESUBMIT = 2     

BANNED_TOKENS = {
    "hamster",
    "jk", "JK",
    "lolita", "Lolita",
    "vtuber", "coser", "cosplay"
}

# -----------------------------
# Utils
# -----------------------------
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

def safe_filename(s: str, max_len: int = 96) -> str:
    s = (s or "").strip()
    s = s.replace("’", "_").replace("'", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if len(s) > max_len else s

def _must_be_ascii(name: str, v: str):
    try:
        (v or "").encode("ascii")
    except Exception:
        raise RuntimeError(
            f"{name} must be ASCII. Found non-ascii characters.\n"
            f"Current {name} repr: {repr(v)}\n"
            f"Fix: export {name}=\"sk-...\" (ASCII only)"
        )

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

def encode_image_to_data_uri(path: str, max_size=(1024, 1024), quality=85) -> str:
    img = open_image_rgb(path)
    img.thumbnail(max_size)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

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

def make_2x2_grid(image_paths: List[str], tile_size: int = GRID_TILE, bg=(255, 255, 255)) -> Optional[Image.Image]:
    valid = []
    for p in image_paths:
        if p and os.path.exists(p):
            valid.append(p)
        if len(valid) >= 4:
            break
    if not valid:
        return None

    canvas = Image.new("RGB", (tile_size * 2, tile_size * 2), bg)
    for i in range(4):
        x0 = (i % 2) * tile_size
        y0 = (i // 2) * tile_size
        if i < len(valid):
            try:
                img = open_image_rgb(valid[i])
                img = _resize_keep_ratio(img, tile_size)
                nx = x0 + (tile_size - img.width) // 2
                ny = y0 + (tile_size - img.height) // 2
                canvas.paste(img, (nx, ny))
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

# -----------------------------
# OpenAI client
# -----------------------------
def get_openai_client() -> OpenAI:
    api_key = (os.environ.get("OPENAI_API_KEY", "") or "").strip()
    base_url = (os.environ.get("OPENAI_BASE_URL", "") or "").strip()

    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in env. e.g. export OPENAI_API_KEY=sk-...")

    _must_be_ascii("OPENAI_API_KEY", api_key)
    if base_url:
        _must_be_ascii("OPENAI_BASE_URL", base_url)

    http_client = httpx.Client(timeout=httpx.Timeout(OPENAI_TIMEOUT_SEC))
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
    return OpenAI(api_key=api_key, http_client=http_client)

client = get_openai_client()

def llm_chat_json(messages: List[Dict[str, Any]], temperature: float, max_retries: int = 3) -> Dict[str, Any]:
    last_err = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
            )
            txt = resp.choices[0].message.content
            return parse_llm_json_output(txt)
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
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("utf-8")
            items.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        except Exception:
            continue
    return items

# -----------------------------
# Dataset loaders & indexes
# -----------------------------
_NOTES_DS = None
_REC_DS = None
_USER_FEAT_DS = None

_USER2RECENT = None
_NOTE_INDEX = None
_USER_FEAT = None

def load_notes_ds():
    global _NOTES_DS
    if _NOTES_DS is None:
        print("Loading notes parquet (dataset handle)...")
        _NOTES_DS = load_dataset("parquet", data_files=NOTES_GLOB, split="train")
        print(_NOTES_DS)
    return _NOTES_DS

def load_rec_ds():
    global _REC_DS
    if _REC_DS is None:
        print("Loading recommendation_train parquet...")
        _REC_DS = load_dataset("parquet", data_files=REC_TRAIN_GLOB, split="train")
        print(_REC_DS)
    return _REC_DS

def load_user_feat_ds():
    global _USER_FEAT_DS
    if _USER_FEAT_DS is None:
        if not os.path.exists(os.path.dirname(USER_FEAT_GLOB)):
            return None
        print("Loading user_feat parquet...")
        _USER_FEAT_DS = load_dataset("parquet", data_files=USER_FEAT_GLOB, split="train")
        print(_USER_FEAT_DS)
    return _USER_FEAT_DS

def build_user2recent() -> Dict[int, List[int]]:
    global _USER2RECENT
    if _USER2RECENT is not None:
        return _USER2RECENT

    if os.path.exists(USER2RECENT_PKL):
        print("Load cached user2recent.pkl")
        with open(USER2RECENT_PKL, "rb") as f:
            _USER2RECENT = pickle.load(f)
        print("user2recent size:", len(_USER2RECENT))
        return _USER2RECENT

    print("No cache. Building user2recent from parquet...")
    ds = load_rec_ds()
    user2recent = {}
    for row in tqdm(ds, desc="build user2recent"):
        uid = int(row["user_idx"])
        lst = row.get("recent_clicked_note_idxs", []) or []
        seen = set()
        dedup = []
        for nid in lst:
            try:
                nid = int(nid)
            except Exception:
                continue
            if nid in seen:
                continue
            seen.add(nid)
            dedup.append(nid)
        if dedup:
            user2recent[uid] = dedup

    with open(USER2RECENT_PKL, "wb") as f:
        pickle.dump(user2recent, f)
    print(" Saved:", USER2RECENT_PKL, "users:", len(user2recent))
    _USER2RECENT = user2recent
    return _USER2RECENT

def _existing_images_abs(image_path_field, limit=MAX_IMAGES_PER_NOTE_INDEX) -> List[str]:
    if not image_path_field:
        return []
    rels = image_path_field if isinstance(image_path_field, list) else [image_path_field]
    out = []
    for rel in rels:
        if not rel:
            continue
        abs_p = os.path.join(IMAGES_ROOT, rel)
        if os.path.exists(abs_p):
            out.append(abs_p)
        if len(out) >= limit:
            break
    return out

def build_note_index_v3() -> Dict[int, Dict[str, Any]]:
    """
   note index:
      note_idx -> {title, content, images[]}
    """
    global _NOTE_INDEX
    if _NOTE_INDEX is not None:
        return _NOTE_INDEX

    if os.path.exists(NOTE_INDEX_PKL):
        print("Load cached note_index.pkl")
        with open(NOTE_INDEX_PKL, "rb") as f:
            _NOTE_INDEX = pickle.load(f)
        print("note_index size:", len(_NOTE_INDEX))
        return _NOTE_INDEX

    print("No cache. Building note_index from notes parquet... (one-time, may take time)")
    ds = load_notes_ds()
    note_idx = {}

    for row in tqdm(ds, desc="build note_index"):
        nid = int(row["note_idx"])
        title = normalize_text(row.get("note_title", ""))
        content = truncate_text(row.get("note_content", ""), MAX_TEXT_CHARS)
        imgs = _existing_images_abs(row.get("image_path", None), limit=MAX_IMAGES_PER_NOTE_INDEX)
        note_idx[nid] = {"title": title, "content": content, "images": imgs}

    with open(NOTE_INDEX_PKL, "wb") as f:
        pickle.dump(note_idx, f)
    print("Saved:", NOTE_INDEX_PKL, "notes:", len(note_idx))
    _NOTE_INDEX = note_idx
    return _NOTE_INDEX

def build_user_feat_index() -> Dict[int, Dict[str, Any]]:
    global _USER_FEAT
    if _USER_FEAT is not None:
        return _USER_FEAT

    if os.path.exists(USER_FEAT_PKL):
        print("Load cached user_feat.pkl")
        with open(USER_FEAT_PKL, "rb") as f:
            _USER_FEAT = pickle.load(f)
        print("user_feat size:", len(_USER_FEAT))
        return _USER_FEAT

    ds = load_user_feat_ds()
    if ds is None:
        _USER_FEAT = {}
        return _USER_FEAT

    idx = {}
    for row in tqdm(ds, desc="build user_feat"):
        try:
            uid = int(row.get("user_idx"))
        except Exception:
            continue
        idx[uid] = {"gender": row.get("gender", None), "age": row.get("age", None)}

    with open(USER_FEAT_PKL, "wb") as f:
        pickle.dump(idx, f)
    print("Saved:", USER_FEAT_PKL, "users:", len(idx))
    _USER_FEAT = idx
    return _USER_FEAT

# -----------------------------
# Product library
# -----------------------------
def _resolve_product_image_path(p: Any) -> Optional[str]:
    if not p:
        return None
    s = str(p).strip()
    if not s:
        return None
    if s.startswith("http://") or s.startswith("https://"):
        return None
    if os.path.exists(s):
        return s
    cand = os.path.join(QILIN_ROOT, s.lstrip("/"))
    if os.path.exists(cand):
        return cand
    return None

def load_product_library() -> List[Dict[str, Any]]:
    if not os.path.exists(PRODUCT_LIBRARY_JSON):
        raise RuntimeError(f"Missing product library: {PRODUCT_LIBRARY_JSON}")
    data = read_json(PRODUCT_LIBRARY_JSON)
    if not isinstance(data, list):
        raise RuntimeError("products_raw.json must be a list")
    for it in data:
        if "product_image" in it:
            it["product_image"] = _resolve_product_image_path(it.get("product_image"))
    print(f"Loaded products: {len(data)}")
    return data

# -----------------------------
# Prompts
# -----------------------------
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

def prompt_pick_rep_images(level: str) -> str:
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
  "chosen": [0, 3, 5, 7],
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

def prompt_storyboard() -> str:
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

def prompt_sora_header() -> str:
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

def _sanitize_banned_tokens(obj: Any) -> Any:
    """
    Remove banned tokens from LLM outputs (extra safety).
    """
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

# -----------------------------
# Profile build (text tiers only) – cached per user
# -----------------------------
def get_text_tiering(uid: int, note_index: Dict[int, Dict[str, Any]], user2recent: Dict[int, List[int]]) -> Dict[str, Any]:
    note_ids = user2recent.get(uid, [])
    if not note_ids:
        raise RuntimeError(f"user {uid} has no recent_clicked_note_idxs")

    note_ids = note_ids[:MAX_NOTES_FOR_TIERING]
    notes_payload = []
    for nid in note_ids:
        meta = note_index.get(int(nid))
        if not meta:
            continue
        notes_payload.append({
            "note_idx": int(nid),
            "title": meta["title"],
            "content": meta["content"],
        })

    if len(notes_payload) < 5:
        raise RuntimeError(f"user {uid} has too few valid notes (<5)")

    prompt = prompt_text_tiering()
    input_json = {"user_id": uid, "notes": notes_payload}
    messages = [{"role": "user", "content": prompt + "\n\nINPUT_JSON:\n" + json.dumps(input_json, ensure_ascii=False)}]
    out = llm_chat_json(messages, temperature=TEMP_TIER)

    out = _sanitize_banned_tokens(out)
    out.setdefault("tiers", {})
    for k in ["strong", "medium", "weak"]:
        out["tiers"].setdefault(k, {"theme": "", "summary": "", "note_idxs": []})
        if "note_idxs" not in out["tiers"][k]:
            out["tiers"][k]["note_idxs"] = []
    out.setdefault("text_style_global", {"tone_tags": [], "info_density": "medium"})
    return out

def build_user_profile(uid: int,
                       note_index: Dict[int, Dict[str, Any]],
                       user2recent: Dict[int, List[int]],
                       user_feat: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    uid = int(uid)
    tiering = get_text_tiering(uid, note_index, user2recent)
    demo = user_feat.get(uid, {})

    profile = {
        "user_id": uid,
        "demographics": {"age": demo.get("age", None), "gender": demo.get("gender", None)},
        "text_preferences": {
            "tiers": tiering.get("tiers", {}),
            "text_style_global": {
                "tone_tags": (tiering.get("text_style_global") or {}).get("tone_tags", []),
                "info_density": (tiering.get("text_style_global") or {}).get("info_density", "medium"),
            }
        }
    }
    out_path = os.path.join(USER_PROFILE_DIR, f"user_{uid}_profile.json")
    write_json(out_path, profile)
    return profile

def load_or_build_user_profile(uid: int,
                               note_index: Dict[int, Dict[str, Any]],
                               user2recent: Dict[int, List[int]],
                               user_feat: Dict[int, Dict[str, Any]],
                               overwrite: bool = False) -> Dict[str, Any]:
    path = os.path.join(USER_PROFILE_DIR, f"user_{uid}_profile.json")
    if (not overwrite) and os.path.exists(path):
        return read_json(path)
    return build_user_profile(uid, note_index, user2recent, user_feat)

# -----------------------------
# Choose tier (strong vs medium only)
# -----------------------------
def choose_best_tier_for_product(user_profile: Dict[str, Any], product: Dict[str, Any]) -> Dict[str, Any]:
    prompt = prompt_choose_tier()
    tiers = (user_profile.get("text_preferences") or {}).get("tiers", {}) or {}

    user_tiers_payload = {
        "strong": {
            "theme": (tiers.get("strong") or {}).get("theme", ""),
            "summary": (tiers.get("strong") or {}).get("summary", ""),
        },
        "medium": {
            "theme": (tiers.get("medium") or {}).get("theme", ""),
            "summary": (tiers.get("medium") or {}).get("summary", ""),
        }
    }

    input_json = {
        "user_tiers": user_tiers_payload,
        "text_style_global": (user_profile.get("text_preferences") or {}).get("text_style_global", {}),
        "product": {
            "product_name": product.get("product_name"),
            "product_slogan": product.get("product_slogan"),
            "product_intro": product.get("product_intro"),
            "product_details": product.get("product_details"),
        }
    }
    messages = [{"role": "user", "content": prompt + "\n\nINPUT_JSON:\n" + json.dumps(input_json, ensure_ascii=False)}]
    out = llm_chat_json(messages, temperature=TEMP_MATCH)
    out = _sanitize_banned_tokens(out)

    if out.get("best_tier") not in ["strong", "medium"]:
        out["best_tier"] = "medium"
    if not isinstance(out.get("confidence"), (int, float)):
        out["confidence"] = 0.6
    if not isinstance(out.get("suggested_hooks"), list):
        out["suggested_hooks"] = []
    return out

# -----------------------------
# Collect tier notes payload (script input)
# -----------------------------
def collect_notes_payload(note_ids: List[int], note_index: Dict[int, Dict[str, Any]], max_n: int = MAX_NOTES_FOR_SCRIPT) -> List[Dict[str, Any]]:
    items = []
    for nid in note_ids[:max_n]:
        meta = note_index.get(int(nid))
        if not meta:
            continue
        items.append({
            "note_idx": int(nid),
            "title": meta["title"],
            "content": meta["content"]
        })
    return items

# -----------------------------
# Pick representative images (same tier only, multi-image)
# -----------------------------
def pick_representative_images_from_tier(level: str,
                                         tier_note_ids: List[int],
                                         note_index: Dict[int, Dict[str, Any]]) -> Tuple[List[str], Dict[str, Any]]:
    """
    Return:
      chosen_paths (<=4)
      debug dict
    """
    candidates: List[str] = []
    for nid in tier_note_ids:
        meta = note_index.get(int(nid))
        if not meta:
            continue
        imgs = meta.get("images", []) or []
        for p in imgs[:MAX_IMAGES_PER_NOTE_FOR_CANDIDATES]:
            if p and os.path.exists(p):
                candidates.append(p)
        if len(candidates) >= MAX_CANDIDATE_IMAGES_TOTAL:
            break

    # unique
    uniq = []
    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    candidates = uniq[:MAX_CANDIDATE_IMAGES_TOTAL]

    debug = {
        "tier": level,
        "tier_note_count": len(tier_note_ids),
        "candidate_count": len(candidates)
    }

    if not candidates:
        debug["chosen_by"] = "none"
        return [], debug

    if len(candidates) <= 4:
        debug["chosen_by"] = "direct_take"
        return candidates, debug

    # use vision LLM to pick up to 4
    prompt = prompt_pick_rep_images(level)
    vision_items = encode_images_for_vision(candidates, max_size=(512, 512), quality=80)

    if len(vision_items) < 4:
        debug["chosen_by"] = "fallback_first4_vision_fail"
        return candidates[:4], debug

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, *vision_items]}]
    out = llm_chat_json(messages, temperature=TEMP_PICK)
    out = _sanitize_banned_tokens(out)

    chosen = out.get("chosen", []) or []
    chosen_idx = []
    used = set()
    for x in chosen:
        try:
            xi = int(x)
        except Exception:
            continue
        if 0 <= xi < len(candidates) and xi not in used:
            used.add(xi)
            chosen_idx.append(xi)
        if len(chosen_idx) >= 4:
            break
    if not chosen_idx:
        chosen_idx = [0, 1, 2, 3]

    chosen_paths = [candidates[i] for i in chosen_idx[:4]]
    debug["chosen_by"] = "llm_pick"
    debug["llm_notes"] = out.get("notes", "")
    return chosen_paths, debug


# -----------------------------
# Build reference images (always in run_dir)
# -----------------------------
def build_reference_images(run_dir: str,
                           rep_paths: List[str],
                           product_img_path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    ensure_dir(run_dir)
    user_grid = make_2x2_grid(rep_paths, tile_size=GRID_TILE)
    if user_grid is None:
        return None, None

    user_grid_path = os.path.join(run_dir, "user_grid.jpg")
    user_grid.save(user_grid_path, "JPEG", quality=92)
    if not product_img_path or not os.path.exists(product_img_path):
        return user_grid_path, user_grid_path
    try:
        product_img = open_image_rgb(product_img_path)
        merged = concat_left_right_keep_height(user_grid, product_img, target_h=SORA_REF_TARGET_H, gap=32)
        sora_ref_path = os.path.join(run_dir, "sora_ref.jpg")
        merged.save(sora_ref_path, "JPEG", quality=92)
        return user_grid_path, sora_ref_path
    except Exception as e:
        print(f"  [Error] Failed to merge images: {e}")
        return user_grid_path, user_grid_path

# -----------------------------
# Extract visual elements
# -----------------------------
def extract_visual_elements_from_grid(user_grid_path: str) -> Dict[str, Any]:
    if not user_grid_path or not os.path.exists(user_grid_path):
        return {"vibe": {"mood_tags": [], "colors_lighting": [], "camera_style": []},
                "elements": {"scenes": [], "objects": [], "actions": [], "composition": []}}

    prompt = prompt_visual_elements()
    vision_items = encode_images_for_vision([user_grid_path], max_size=(768, 768), quality=85)
    if not vision_items:
        return {"vibe": {"mood_tags": [], "colors_lighting": [], "camera_style": []},
                "elements": {"scenes": [], "objects": [], "actions": [], "composition": []}}

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, *vision_items]}]
    out = llm_chat_json(messages, temperature=TEMP_VTOK)
    out = _sanitize_banned_tokens(out)

    out.setdefault("vibe", {})
    out.setdefault("elements", {})
    for k in ["mood_tags", "colors_lighting", "camera_style"]:
        out["vibe"].setdefault(k, [])
        if not isinstance(out["vibe"][k], list):
            out["vibe"][k] = []
    for k in ["scenes", "objects", "actions", "composition"]:
        out["elements"].setdefault(k, [])
        if not isinstance(out["elements"][k], list):
            out["elements"][k] = []
    return out


def upload_to_aliyun_oss(local_file_path: str) -> Optional[str]:
    if not local_file_path or not os.path.exists(local_file_path):
        return None
    if not OSS_ACCESS_KEY_ID or not OSS_ACCESS_KEY_SECRET:
        print("  [OSS Error] Missing Aliyun Access Keys in environment variables!")
        return None

    try:
        auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
        bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET_NAME)

        ext = os.path.splitext(local_file_path)[-1]
        remote_name = f"sora_ref/{uuid.uuid4()}{ext}"

        print(f"  [OSS] Uploading reference to Aliyun: {remote_name}")
        with open(local_file_path, 'rb') as fileobj:
            result = bucket.put_object(remote_name, fileobj)
        
        if result.status == 200:
            return f"https://{OSS_BUCKET_NAME}.{OSS_ENDPOINT}/{remote_name}"
        return None
    except Exception as e:
        print(f"  [OSS Error] {e}")
        return None

# -----------------------------
# Storyboard
# -----------------------------
def generate_storyboard(user_profile: Dict[str, Any],
                        product: Dict[str, Any],
                        match_result: Dict[str, Any],
                        tier_notes_payload: List[Dict[str, Any]],
                        visual: Dict[str, Any]) -> Dict[str, Any]:
    prompt = prompt_storyboard()
    demo = user_profile.get("demographics", {})
    text_style = (user_profile.get("text_preferences") or {}).get("text_style_global", {})
    tiers = (user_profile.get("text_preferences") or {}).get("tiers", {})
    best_tier = match_result.get("best_tier", "medium")

    tier_obj = (tiers.get(best_tier) or {})
    tier_summary = tier_obj.get("summary", "")
    tier_theme = tier_obj.get("theme", "")

    input_json = {
        "user_demographics": {"age": demo.get("age", "unknown"), "gender": demo.get("gender", "unknown")},
        "text_style_global": {"tone_tags": text_style.get("tone_tags", []), "info_density": text_style.get("info_density", "medium")},
        "tier_used": {"tier": best_tier, "theme": tier_theme, "summary": tier_summary, "tier_notes": tier_notes_payload},
        "visual": visual,
        "product": {
            "product_name": product.get("product_name"),
            "product_slogan": product.get("product_slogan"),
            "product_intro": product.get("product_intro"),
            "product_details": product.get("product_details"),
        },
        "suggested_hooks": match_result.get("suggested_hooks", [])
    }

    messages = [{"role": "user", "content": prompt + "\n\nINPUT_JSON:\n" + json.dumps(input_json, ensure_ascii=False)}]
    out = llm_chat_json(messages, temperature=TEMP_STORY)
    out = _sanitize_banned_tokens(out)

    out.setdefault("duration_sec", 15)
    out.setdefault("format", "9:16")
    out.setdefault("tier_used", best_tier)
    return out

def build_sora_web_prompt(storyboard: Dict[str, Any]) -> str:
    header = prompt_sora_header()
    story_txt = json.dumps(storyboard, ensure_ascii=False, indent=2)
    return header + "\n\nSTORYBOARD_JSON:\n" + story_txt + "\n"


def apimart_token() -> str:
    tok = (
        (os.environ.get("APIMART_TOKEN", "") or "").strip()
        or (os.environ.get("SORA_API_KEY", "") or "").strip()
        or (os.environ.get("SORA2_API_KEY", "") or "").strip()
    )
    if not tok:
        raise RuntimeError("Missing APIMART_TOKEN (or SORA_API_KEY / SORA2_API_KEY) in env")
    _must_be_ascii("APIMART_TOKEN", tok)
    return tok

def sora2_submit(prompt_text: str, ref_image_path: Optional[str], duration: int, aspect_ratio: str, model: str) -> str:
    url = "https://api.apimart.ai/v1/videos/generations"
    headers = {"Authorization": f"Bearer {apimart_token()}", "Content-Type": "application/json"}

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt_text,
        "duration": int(duration),
        "aspect_ratio": aspect_ratio,
        "watermark": bool(SORA2_WATERMARK),
    }

    if ref_image_path and os.path.exists(ref_image_path):
        public_url = upload_to_aliyun_oss(ref_image_path)
        
        if public_url:
            print(f"  [Sora API] Using reference URL: {public_url}")
            payload["image_urls"] = [public_url]
        else:
            print(f"  [Sora API] Warning: Failed to get OSS URL, submitting without image.")

    r = requests.post(url, json=payload, headers=headers, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    if j.get("code") != 200:
        raise RuntimeError(f"Apimart submit failed: {j}")
    data = j.get("data", [])
    if not data or not data[0].get("task_id"):
        raise RuntimeError(f"Apimart submit no task_id: {j}")
    return data[0]["task_id"]

def sora2_query(task_id: str) -> Dict[str, Any]:
    url = f"https://api.apimart.ai/v1/tasks/{task_id}?language=en"
    headers = {"Authorization": f"Bearer {apimart_token()}"}
    r = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    if j.get("code") != 200:
        raise RuntimeError(f"Apimart poll failed: {j}")
    return j.get("data", {}) or {}

def sora2_poll_until_video(task_id: str,
                          max_sec: int = POLL_MAX_SEC,
                          base_interval: int = POLL_BASE_INTERVAL) -> Tuple[str, Dict[str, Any]]:
    t0 = time.time()
    interval = base_interval
    error_count = 0
    last_data = {}

    while True:
        if time.time() - t0 > max_sec:
            raise TimeoutError(f"Polling timeout for task={task_id} after {max_sec}s")

        try:
            data = sora2_query(task_id)
            last_data = data
            status = (data.get("status") or "").lower()

            if status == "completed":
                videos = (((data.get("result") or {}).get("videos")) or [])
                if videos:
                    v0 = videos[0]
                    if isinstance(v0, dict) and v0.get("url"):
                        u = v0["url"]
                        if isinstance(u, list) and u:
                            return u[0], last_data
                        if isinstance(u, str):
                            return u, last_data
                raise RuntimeError(f"completed but no video url: {data}")

            if status in ("failed", "error", "cancelled"):
                raise RuntimeError(f"task failed: status={status}, data={data}")

            # keep polling
            time.sleep(interval)
            interval = min(int(interval * 1.2), 45)

        except Exception as e:
            error_count += 1
            time.sleep(min(8 + error_count * 2, 60))
            if error_count >= MAX_POLL_ERRORS_BEFORE_RESUBMIT:
                raise RuntimeError(f"too many poll errors for task={task_id}: {e}")

def download_file_with_retry(url: str, out_path: str, max_retries: int = 6) -> None:
    ensure_dir(os.path.dirname(out_path))
    last_err = None
    for i in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=HTTP_TIMEOUT * 3) as r:
                r.raise_for_status()
                tmp = out_path + ".tmp"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp, out_path)
                return
        except Exception as e:
            last_err = e
            time.sleep(3 + i * 2)
    raise RuntimeError(f"download failed after retries: {last_err}")

# -----------------------------
# Progress & run
# -----------------------------
def progress_path() -> str:
    return os.path.join(RUNS_DIR, "progress.json")

def load_progress() -> Dict[str, Any]:
    p = progress_path()
    if os.path.exists(p):
        return read_json(p)
    return {"runs": {}}

def save_progress(prog: Dict[str, Any]):
    write_json(progress_path(), prog)


def local_pack_copy(run_dir: str, local_out: str, uid: int, product_safe: str):
    if not local_out:
        return
    user_safe = f"user_{uid}"
    local_user_dir = os.path.join(local_out, user_safe, product_safe)
    ensure_dir(local_user_dir)

    want_files = [
        "video.mp4",
        f"video__user_{uid}__{product_safe}.mp4",
        "sora_ref.jpg",
        "user_grid.jpg",
        "sora_web_prompt.txt",
        "review.json",
        "storyboard.json",
        "tier_notes_used.json",
        "visual_elements.json",
        "match_result.json",
        "rep_images.json",
        "product_preview.jpg",
        "sora_task.json",
    ]

    for fn in want_files:
        src = os.path.join(run_dir, fn)
        if os.path.exists(src):
            dst = os.path.join(local_user_dir, fn)
            with open(src, "rb") as rf, open(dst, "wb") as wf:
                wf.write(rf.read())

def run_one_prepare_and_maybe_submit(uid: int,
                                    product: Dict[str, Any],
                                    user_profile: Dict[str, Any],
                                    note_index: Dict[int, Dict[str, Any]],
                                    do_sora: bool,
                                    overwrite: bool,
                                    submit_only: bool,
                                    prog: Dict[str, Any]) -> None:

    product_name = product.get("product_name", "unknown_product")
    product_safe = safe_filename(product_name)
    user_safe = f"user_{uid}"
    run_dir = os.path.join(RUNS_DIR, user_safe, product_safe)
    ensure_dir(run_dir)

    run_key = f"{uid}__{product_safe}"
    prog.setdefault("runs", {})
    prog["runs"].setdefault(run_key, {"status": "init", "resubmits": 0})

    storyboard_path = os.path.join(run_dir, "storyboard.json")
    sora_web_prompt_path = os.path.join(run_dir, "sora_web_prompt.txt")
    video_path = os.path.join(run_dir, "video.mp4")
    task_meta_path = os.path.join(run_dir, "sora_task.json")

    if (not overwrite) and os.path.exists(storyboard_path) and os.path.exists(sora_web_prompt_path):
        if (not do_sora) or os.path.exists(video_path) or (submit_only and os.path.exists(task_meta_path)):
            prog["runs"][run_key]["status"] = "done_or_ready"
            save_progress(prog)
            print(f"Skip prepared: user={uid}, product={product_name}")
            return

    print(f"\n---- user={uid}, product={product_name} ----")
    print("  [1/8] Matching best tier (strong vs medium)...")
    match_result = choose_best_tier_for_product(user_profile, product)
    write_json(os.path.join(run_dir, "match_result.json"), match_result)
    best_tier = match_result["best_tier"]
    tiers = (user_profile.get("text_preferences") or {}).get("tiers", {}) or {}
    tier_note_ids = (tiers.get(best_tier, {}) or {}).get("note_idxs", []) or []
    tier_note_ids = [int(x) for x in tier_note_ids if str(x).strip().isdigit()]
    print(f"  [2/8] Picking 4 representative images from best_tier={best_tier} (no cross-tier)...")
    chosen_paths, pick_debug = pick_representative_images_from_tier(best_tier, tier_note_ids, note_index)
    write_json(os.path.join(run_dir, "rep_images.json"), {"chosen_paths": chosen_paths, "debug": pick_debug})
    print("  [3/8] Building reference images...")
    product_img_path = product.get("product_image", None)
    user_grid_path, sora_ref_path = build_reference_images(run_dir, chosen_paths, product_img_path)
    try:
        if product_img_path and os.path.exists(product_img_path):
            img = open_image_rgb(product_img_path)
            img = _resize_keep_ratio(img, 768)
            img.save(os.path.join(run_dir, "product_preview.jpg"), "JPEG", quality=90)
    except Exception:
        pass

    if not chosen_paths:
        print("No representative images found for best_tier (this tier may have no valid image files).")
    if not user_grid_path:
        print(" No user_grid. Continue with text-only visual.")
    if not sora_ref_path:
        print(" No sora_ref (missing product image or grid). Sora may be skipped.")

    print("  [4/8] Extracting visual elements (vibe + concrete elements)...")
    if user_grid_path and os.path.exists(user_grid_path):
        visual = extract_visual_elements_from_grid(user_grid_path)
    else:
        visual = {"vibe": {"mood_tags": [], "colors_lighting": [], "camera_style": []},
                  "elements": {"scenes": [], "objects": [], "actions": [], "composition": []}}
    write_json(os.path.join(run_dir, "visual_elements.json"), visual)

    print("  [5/8] Preparing tier notes payload...")
    tier_notes_payload = collect_notes_payload(tier_note_ids, note_index, max_n=MAX_NOTES_FOR_SCRIPT)
    write_json(os.path.join(run_dir, "tier_notes_used.json"), tier_notes_payload)

    print("  [6/8] Generating storyboard (anti-static, last 2s CTA)...")
    storyboard = generate_storyboard(user_profile, product, match_result, tier_notes_payload, visual)
    write_json(storyboard_path, storyboard)

    print("  [7/8] Writing sora_web_prompt.txt ...")
    sora_web_prompt = build_sora_web_prompt(storyboard)
    with open(sora_web_prompt_path, "w", encoding="utf-8") as f:
        f.write(sora_web_prompt)

    # review.json
    review = {
        "user_id": uid,
        "product_name": product_name,
        "best_tier": best_tier,
        "confidence": match_result.get("confidence", None),
        "files": {
            "match_result": "match_result.json",
            "rep_images": "rep_images.json",
            "user_grid": "user_grid.jpg" if user_grid_path and os.path.exists(user_grid_path) else None,
            "sora_ref": "sora_ref.jpg" if sora_ref_path and os.path.exists(sora_ref_path) else None,
            "visual_elements": "visual_elements.json",
            "tier_notes_used": "tier_notes_used.json",
            "storyboard": "storyboard.json",
            "sora_web_prompt": "sora_web_prompt.txt",
            "video_mp4": "video.mp4" if os.path.exists(os.path.join(run_dir, "video.mp4")) else None,
            "sora_task": "sora_task.json" if os.path.exists(task_meta_path) else None
        }
    }
    write_json(os.path.join(run_dir, "review.json"), review)

    prog["runs"][run_key]["status"] = "storyboard_ready"
    save_progress(prog)

    print("  [8/8] (optional) Sora2 submit (no wait)...")
    if not do_sora:
        return

    if not sora_ref_path or not os.path.exists(sora_ref_path):
        print(f"No sora_ref. Skip submit for user={uid}, product={product_name}")
        prog["runs"][run_key]["status"] = "skip_sora_no_ref"
        save_progress(prog)
        return

    if os.path.exists(os.path.join(run_dir, "video.mp4")) and (not overwrite):
        prog["runs"][run_key]["status"] = "done"
        save_progress(prog)
        return

    task_meta = {}
    if os.path.exists(task_meta_path):
        try:
            task_meta = read_json(task_meta_path) or {}
        except Exception:
            task_meta = {}

    existing_task = task_meta.get("task_id")
    existing_status = (task_meta.get("status") or "").lower()
    if existing_task and existing_status not in ("failed", "error", "cancelled"):
        print(f"Existing task found, skip submit: task_id={existing_task}, status={existing_status}")
        prog["runs"][run_key]["status"] = "submitted"
        save_progress(prog)
        return

    # submit
    task_id = sora2_submit(
        prompt_text=sora_web_prompt,
        ref_image_path=sora_ref_path,
        duration=SORA2_DURATION,
        aspect_ratio=SORA2_ASPECT,
        model=SORA2_MODEL
    )
    task_meta = {
        "task_id": task_id,
        "status": "submitted",
        "created_at": time.time(),
        "poll_errors": 0,
        "resubmits": int(task_meta.get("resubmits", 0)) if isinstance(task_meta, dict) else 0,
    }
    write_json(task_meta_path, task_meta)
    print("Submitted Sora2 task:", task_id)

    prog["runs"][run_key]["status"] = "submitted"
    save_progress(prog)

def _load_task_meta(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        return read_json(path) or {}
    except Exception:
        return {}

def _save_task_meta(path: str, meta: Dict[str, Any]):
    write_json(path, meta)

def poll_download_one_job(job: Dict[str, Any]) -> Dict[str, Any]:
    run_dir = job["run_dir"]
    uid = job["uid"]
    product_safe = job["product_safe"]
    product_name = job["product_name"]
    local_out = job.get("local_out", "")

    sora_web_prompt_path = os.path.join(run_dir, "sora_web_prompt.txt")
    sora_ref_path = os.path.join(run_dir, "sora_ref.jpg")
    task_meta_path = os.path.join(run_dir, "sora_task.json")
    video_path = os.path.join(run_dir, "video.mp4")
    video_named = os.path.join(run_dir, f"video__user_{uid}__{product_safe}.mp4")

    if os.path.exists(video_path):

        return {"status": "already_done", "uid": uid, "product": product_name}

    if (not os.path.exists(sora_web_prompt_path)) or (not os.path.exists(sora_ref_path)):
        return {"status": "missing_prompt_or_ref", "uid": uid, "product": product_name}

    with open(sora_web_prompt_path, "r", encoding="utf-8") as f:
        sora_web_prompt = f.read()

    task_meta = _load_task_meta(task_meta_path)
    resubmits = int(task_meta.get("resubmits", 0)) if isinstance(task_meta, dict) else 0

    def do_submit_new_task(reason: str) -> str:
        nonlocal resubmits, task_meta
        task_id = sora2_submit(
            prompt_text=sora_web_prompt,
            ref_image_path=sora_ref_path,
            duration=SORA2_DURATION,
            aspect_ratio=SORA2_ASPECT,
            model=SORA2_MODEL
        )
        resubmits = resubmits + 1
        task_meta = {
            "task_id": task_id,
            "status": "submitted",
            "created_at": time.time(),
            "poll_errors": 0,
            "resubmits": resubmits,
            "resubmit_reason": reason,
        }
        _save_task_meta(task_meta_path, task_meta)
        return task_id

    task_id = task_meta.get("task_id")
    status = (task_meta.get("status") or "").lower()

    if (not task_id) or (status in ("failed", "error", "cancelled")):
        if resubmits >= MAX_RESUBMIT:
            return {"status": "give_up_no_task", "uid": uid, "product": product_name}
        task_id = do_submit_new_task(reason=f"no_task_or_bad_status({status})")

    while True:
        try:
            # poll
            task_meta["status"] = "polling"
            task_meta["last_polled_at"] = time.time()
            _save_task_meta(task_meta_path, task_meta)

            video_url, last_data = sora2_poll_until_video(task_id)
            task_meta["status"] = "completed"
            task_meta["video_url"] = video_url
            task_meta["last_data"] = last_data
            task_meta["completed_at"] = time.time()
            _save_task_meta(task_meta_path, task_meta)

            download_file_with_retry(video_url, video_path)

            try:
                if os.path.exists(video_path) and (not os.path.exists(video_named)):
                    with open(video_path, "rb") as rf, open(video_named, "wb") as wf:
                        wf.write(rf.read())
            except Exception:
                pass

            if local_out:
                local_pack_copy(run_dir, local_out, uid, product_safe)

            return {"status": "done", "uid": uid, "product": product_name, "task_id": task_id}

        except Exception as e:
            task_meta["status"] = "poll_or_download_error"
            task_meta["error"] = str(e)
            task_meta["last_error_at"] = time.time()
            task_meta["poll_errors"] = int(task_meta.get("poll_errors", 0)) + 1
            _save_task_meta(task_meta_path, task_meta)

            pe = int(task_meta.get("poll_errors", 0))
            if pe < MAX_POLL_ERRORS_BEFORE_RESUBMIT:

                time.sleep(10)
                continue

            if resubmits >= MAX_RESUBMIT:
                return {"status": "failed_max_resubmit", "uid": uid, "product": product_name, "error": str(e)}

            task_meta["status"] = "resubmit_due_to_errors"
            _save_task_meta(task_meta_path, task_meta)

            task_id = do_submit_new_task(reason=f"too_many_poll_errors({pe})")

def collect_pending_jobs(uids: List[int],
                         products: List[Dict[str, Any]],
                         only_these: Optional[List[Tuple[int, str]]] = None,
                         local_out: str = "") -> List[Dict[str, Any]]:

    jobs = []
    for uid in uids:
        user_safe = f"user_{uid}"
        user_dir = os.path.join(RUNS_DIR, user_safe)
        if not os.path.exists(user_dir):
            continue

        for p in products:
            product_name = p.get("product_name", "unknown_product")
            product_safe = safe_filename(product_name)
            run_dir = os.path.join(user_dir, product_safe)
            if not os.path.exists(run_dir):
                continue

            if os.path.exists(os.path.join(run_dir, "video.mp4")):
                continue

            task_meta_path = os.path.join(run_dir, "sora_task.json")
            sora_ref_path = os.path.join(run_dir, "sora_ref.jpg")
            sora_prompt_path = os.path.join(run_dir, "sora_web_prompt.txt")

            if not (os.path.exists(task_meta_path) and os.path.exists(sora_ref_path) and os.path.exists(sora_prompt_path)):
                continue

            task_meta = _load_task_meta(task_meta_path)
            task_id = task_meta.get("task_id")
            if not task_id:
                continue

            jobs.append({
                "run_dir": run_dir,
                "uid": uid,
                "product_safe": product_safe,
                "product_name": product_name,
                "local_out": local_out
            })

    return jobs

def poll_download_jobs_concurrently(jobs: List[Dict[str, Any]], concurrency: int = 4):
    if not jobs:
        print("No pending jobs to poll/download.")
        return

    print(f"Start concurrent poll+download: jobs={len(jobs)}, concurrency={concurrency}")
    done = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(poll_download_one_job, j) for j in jobs]
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                failed += 1
                print(f"worker crashed: {e}")
                continue

            st = r.get("status")
            if st in ("done", "already_done"):
                done += 1
                print(f"[{done}/{len(jobs)}] done: user={r.get('uid')}, product={r.get('product')}")
            else:
                failed += 1
                print(f"[{done}/{len(jobs)}] not done: status={st}, user={r.get('uid')}, product={r.get('product')}, err={r.get('error')}")

    print(f"Poll+download finished. done={done}, failed={failed}")

def parse_users_arg(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    parts = re.split(r"[,\s]+", s)
    out = []
    for p in parts:
        if not p:
            continue
        out.append(int(p))
    return out


def main():
    print("Starting pipeline A v3 (batch submit + concurrent download) ...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=str, required=True,
                        help="User list, e.g. '11094,15067,576'")
    parser.add_argument("--products", type=str, default="ALL",
                        help="Product names separated by '||'. Default ALL.")
    parser.add_argument("--do_sora", action="store_true", help="Call Sora2 API and download mp4")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite cached results")
    parser.add_argument("--local_out", type=str, default="",
                        help="Optional: directory to copy mp4/ref/prompt for easy viewing")

    # v3 new controls
    parser.add_argument("--sora_concurrency", type=int, default=4,
                        help="Concurrent workers for poll+download (default 4).")
    parser.add_argument("--submit_only", action="store_true",
                        help="Only submit tasks (no polling/downloading).")
    parser.add_argument("--poll_only", action="store_true",
                        help="Only poll+download pending tasks (no new submit).")

    args = parser.parse_args()

    uids = parse_users_arg(args.users)
    if not uids:
        raise RuntimeError("No users parsed from --users")

    products = load_product_library()
    if args.products != "ALL":
        wanted = [x.strip() for x in args.products.split("||") if x.strip()]
        wanted_set = set(wanted)
        products = [p for p in products if p.get("product_name") in wanted_set]
        print(f"Filtered products: {len(products)}")
    else:
        print(f"Using ALL products: {len(products)}")

    user2recent = build_user2recent()
    note_index = build_note_index()
    user_feat = build_user_feat_index()

    prog = load_progress()
    local_out = args.local_out.strip() if args.local_out else ""

    if args.poll_only:
        if not args.do_sora:
            print("You used --poll_only but did not set --do_sora. Still polling based on existing tasks.")
        jobs = collect_pending_jobs(uids, products, local_out=local_out)
        poll_download_jobs_concurrently(jobs, concurrency=max(1, int(args.sora_concurrency)))
        print("\n Poll-only finished.")
        print("Results dir:", RUNS_DIR)
        return


    for uid in uids:
        if uid not in user2recent:
            print(f"user {uid} has no history in recommendation_train; skip")
            continue

        user_profile = load_or_build_user_profile(
            uid=uid,
            note_index=note_index,
            user2recent=user2recent,
            user_feat=user_feat,
            overwrite=args.overwrite
        )

        for product in products:
            tries = 0
            while True:
                tries += 1
                try:
                    run_one_prepare_and_maybe_submit(
                        uid=uid,
                        product=product,
                        user_profile=user_profile,
                        note_index=note_index,
                        do_sora=args.do_sora,
                        overwrite=args.overwrite,
                        submit_only=args.submit_only,
                        prog=prog
                    )
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"error in user={uid}, product={product.get('product_name')}: {e}")
                    if tries >= 3:
                        print("giving up after 3 tries, continue next")
                        break
                    time.sleep(5)

    if args.submit_only:
        print("\n Submit-only finished. (tasks submitted, no polling/downloading)")
        print("Results dir:", RUNS_DIR)
        return

    if args.do_sora:
        jobs = collect_pending_jobs(uids, products, local_out=local_out)
        poll_download_jobs_concurrently(jobs, concurrency=max(1, int(args.sora_concurrency)))

    print("\n Batch finished.")
    print("Results dir:", RUNS_DIR)
    print("Progress:", progress_path())

if __name__ == "__main__":
    main()
