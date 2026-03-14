import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
from openai import OpenAI
import os
import json
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

prompt = '''**Role:** You are an expert Video Content Strategist and Media Analyst. Your task is to dissect short-form videos into a structured "Style Code" that captures the creator's content DNA and presentation signature.

**Objective:** Analyze the provided video (visual frames) and categorize it according to the specific taxonomy below. Your analysis will be used to measure the consistency of a creator's persona.

------

### 1. Taxonomy & Dimensions (The Style Code)

**A. Content Topic (What is it about?)**

- `[Tech/Digital, Food/Cooking, Travel/Outdoor, Daily Life/Vlog, Gaming, Film/Drama Commentary, Fitness/Health, Beauty/Fashion, Educational/Facts, Music, Pet, Other]`

**B. Visual Presentation (How does it look?)**

- **Tone:** `[Warm/Healing, High Saturation/Vibrant, Low Saturation/Minimalist, Cinematic/Moody, Anime/Stylized]`
- **Camera Work:** `[Static/Tripod, Dynamic/Handheld, Aerial/Drone, First-Person-Perspective (POV), Macro-Focused]`

**C. Narrative Template (How is it structured?)**

- `[Tutorial/How-to, Journal-style Vlog, Scripted/Plot-Twist, Professional Review, Aesthetic/Music Montage]`

**D. Linguistic Tone (How does it sound?)**

- `[Soothing/Emotional, Formal/Professional, Humorous/Witty, Sharp/Snarky, Energetic/Hype, Music-Only]`

**E. Information Density (What is the value?)**

- `[Hardcore/Fact-Heavy, Emotional/Atmospheric, Casual/Light-Entertainment]`

------

### 2. Constraints & Output Format

1. **Output Format:** Return ONLY a valid **JSON** object.
2. **Strict Selection:** Choose the closest match from the provided options. If none apply, use `Other`.

------

### 3. Expected JSON Structure

```json
{
  "topic": "Tech/Digital",
  "presentation": {
    "tone": "Cinematic/Moody",
    "camera_work": "Macro-Focused"
  },
  "narrative_template": "Professional Review",
  "linguistic_tone": "Formal/Professional",
  "info_density": "Hardcore/Fact-Heavy"
}
```'''

def parse_response(response):
    try:
        json_str = response.split("```json")[1].split("```")[0].strip('\n').strip()
        return json.loads(json_str, strict=False)
    except:
        return None

def process_frames(vid):
    video = cv2.VideoCapture(f"all_videos/{vid}.mp4")
    base64Frames = []
    frame_idx = 0
    while video.isOpened():
        if frame_idx < 1000:
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
            frame_idx += 1
        else:
            break

    video.release()
    return base64Frames

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL")
)

def process_one_video(vid):

    frames = process_frames(vid)
    if len(frames) == 0:
        return vid, None

    response = None
    for trial in range(3):
        try:
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": prompt
                            },
                            *[
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{frame}"
                                }
                                for frame in frames[0::50]
                            ]
                        ]
                    }
                ],
            )
            if response:
                break
        except Exception:
            time.sleep(3)

    if response:
        try:
            res = parse_response(response.output_text)
            return vid, res
        except Exception:
            return vid, None

    return vid, None


def main():
    with open('users.json', 'r') as f:
        users = json.load(f)

    vids = []
    for user in users:
        for vid in user['vids']:
            if vid not in vids:
                vids.append(vid)

    results = {}

    max_workers = 64

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_one_video, vid): vid
            for vid in vids
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            vid, res = future.result()
            if res is not None:
                results[vid] = res
    
    from collections import Counter

    def get_preference(sequence):
        dimensions = {
            'topic': 0.25,
            'narrative_template': 0.20,
            'linguistic_tone': 0.10,
            'info_density': 0.10,
            'tone': 0.20,
            'camera_work': 0.15
        }
        
        preference = {
            "presentation": {}
        }
        
        for dim, weight in dimensions.items():
            if dim in ['tone', 'camera_work']:
                tags = [item['presentation'][dim] for item in sequence]
            else:
                tags = [item[dim] for item in sequence]
                
            most_common_item = Counter(tags).most_common(1)[0]
            mode_tag = most_common_item[0]
            
            dimension_consistency = mode_count / len(sequence)
            
            
            if dim in ['tone', 'camera_work']:
                preference["presentation"][dim] = mode_tag
            else:
                preference[dim] = mode_tag
        
        return preference
    
    for user in users:
        user['preference'] = get_preference([results[vid] for vid in user['vids']])
    
    with open("users_processed.json", "w") as f:
        json.dump(users, f, indent=4)

if __name__ == "__main__":
    main()

