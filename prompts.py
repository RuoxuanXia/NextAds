video_summary_prompt = '''# Role
You are a Professional Video Content Analyst & Advertising Strategist.

# Task
Analyze the target video and provide a structured breakdown to facilitate "Native Ad Script" generation. The goal is to ensure the ad script matches the video's original vibe, rhythm, and logic.

# Output Format (Strict JSON)
```json
{
  "overall_topic": "Categorized Topic: Specific Sub-topic (e.g., Gaming: Honor of Kings Guide)",
  "content_summary": "A 2-3 sentence detailed description of the narrative flow and key events.",
  "visual_aesthetic": {
    "style_tags": ["list of tags like Cinematic, Minimalist, etc."],
    "color_grading": "Description of the dominant color palette",
    "lighting": "Description of the lighting quality"
  },
  "audio_environment": {
    "type": "BGM only / Voiceover / ASMR / Ambient",
    "mood": "Emotional tone of the audio",
    "pacing_bpm": "Fast / Moderate / Slow"
  },
  "narrative_logic": {
    "structure": "e.g., Linear tutorial, Random Vlog, High-energy montage",
    "pacing_score": "1-10 (1 is slow/serene, 10 is fast/chaotic)"
  }
}
```
'''

preference_summary_prompt = '''# Role
You are a Senior Visual Preference Analyst & Video Aesthetic Researcher. Your expertise lies in synthesizing structured data with visual evidence to decode a user's "Fine-Grained Visual DNA."

# Input Data
1. **Coarse-Grained Preferences:** [Insert Rule-based Results here, e.g., Topic: Food, Tone: High Saturation, Template: Tutorial]
2. **Visual Keyframes:** A sequence of images representing the user's most consistent content.

# Task
Analyze the provided images in the context of the coarse-grained metadata. Your goal is to move beyond general labels and identify the specific, nuanced aesthetic choices that define this user.

# Strict Constraints
1. **The 70% Consistency Rule:** Only include a trait in the "dominant" categories if it appears in ≥ 70% of the visual samples.
2. **Professional Precision:** Use specific industry terminology (e.g., "Flat Lay," "Tonal Compression," "Soft Box Lighting," "Earthy Minimalism").
3. **No Conversational Filler:** Output the result strictly in the specified JSON format.

---

# Strict Instructions for Specificity:

1. **Name the Entities:** Do not just say "Gaming"; name the specific game if visible (e.g., "Honor of Kings," "League of Legends").
2. **Identify the Sub-niche:** Do not just say "Food"; specify the cuisine or setting (e.g., "Chongqing Spicy Hotpot," "Street Food Vlogs," "Fine Dining").
3. **Identify the Species:** Do not just say "Pets"; specify "Ginger Cats," "Golden Retrievers," etc.
4. **Contextual Description:** Describe the specific activity (e.g., "Character skill tutorials," "ASMR eating sounds," "Funny animal fails")..

---

# Input Preference
Topic: {topic}
Tone: {tone}
Camera Work: {camera_work}
Narrative Template: {narrative_template}

# Final Output Format (Strict JSON)

```json
{{
    "content_preference": "Must be concrete. Example: 'Gaming commentary for Honor of Kings and cinematic vlogs of Chongqing street food spicy hotpots.'",
    "style_preference": "Must be descriptive. Example: 'Bright, high-saturation POV shots with fast-paced editing and vibrant on-screen captions.'",
    "detailed_elements": "List specific elements that users prefer, such as game characters, food content, film/TV styles, road trips, Africa scenes."
}}
```'''

insertion_frame_prompt = '''# Role
You are an AI Video Editor & Ad Placement Strategist. Your goal is to analyze a target video's second-by-second content and determine the most "native" timestamp to insert a 15-second advertisement.

# Task
Identify the optimal "Ad Break" ($T_{{insert}}$) where the product can be introduced with minimum disruption and maximum thematic synergy.

# Inputs
1. **Target Video Timeline:**
2. **Product Information:** 
- Product name: {product_name}
- Product details: {product_details}

# Placement Criteria (Logic)
1. **Narrative Pause:** Look for scene transitions, the end of a sentence, or a moment of lower visual/audio intensity.
2. **Thematic Bridge:** Is there a second where the video mentions a "problem" that the product "solves"?
3. **Emotional Alignment:** Match the ad's entry to the video's emotional lull or a logical transition point (e.g., after a "How-to" step is completed).

# Output Format (Strict JSON)
Return ONLY a JSON object with the following structure:
The insertion_timestamp means the insert second. For example, if you think we can add an advertisement in the 6th second (after the 6th frame/6th input picture), you just answer 6.

{{
  "insertion_timestamp": 7,
  "insertion_logic": "Explain why this second is the best choice (e.g., 'Scene transition from kitchen to dining area').",
  "bridge_description": "The video mentions [X] at second 6, which perfectly leads into the product [Y].",
  "visual_cue_to_trigger": "What visual element should we use to transition (e.g., a fade-out or a zoom-in)?"
}}'''

script_generation_prompt = '''
You are a short-form ad director and copywriter.

We need to insert a product placement within the target video. You need to start by designing how to introduce the product within the target video's content, aligning it with user preferences, from the very first frame.

Create a 15-second 16:9 in-feed native ad storyboard.
The ad must feel like organic content and must NOT be static.
The ad must align with user preferences, including the content, style, and elements that users like.
The ad must be strongly relevant to the content of the current video frame.

Inputs:

User Preference:
- content_preference: {content_preference}
- style_preference {style_preference}
- detailed elements {detailed_elements}

Current Video Summary:
- overall_content: {overall_content}
- content_summary: {content_summary}

Product Information:
- product_name: {product_name}
- product_slogan: {product_slogan}
- product_intro: {product_intro}
- product_details: {product_details}

Two reference figures:
- the picture of the product
- the picture of the start frame of the target video

Hard constraints:
1) Must have motion: every scene must include camera motion OR subject motion.
2) Must change at least every 2-3 seconds (new shot, new action, new angle).
3) **Start from the given start frame (the second given image, with description in the scene) and return to the given start frame at the final 2 seconds**
4) Visual personalization must be grounded in provided visual elements (scenes/objects/actions/colors).
5) Product depiction must be faithful; no wrong logos; no shape distortion.
6) We do not need a clear CTA for the product, this is a soft advertisement.
7) Everything in the ad must be in English.

Integration Rule:
- The product must be introduced as a natural object already existing in the target video’s world,
  not as an external promotion.
- Every product appearance must be justified by the ongoing action, environment, or intent
  shown in the target video.
- If the product were removed, the scene should still make logical sense.

Fallback Integration (use only if direct in-world integration is not plausible):
- The target video may be reframed as content playing on a phone or computer screen.
- The camera can pull back into a real-life environment that naturally fits the product,
  then introduce the product through that context.

Output ENGLISH JSON ONLY:
```json
{{
  "duration_sec": 15,
  "format": "16:9",
  "hook": "one sentence hook",
  "scenes": [
    {{
      "sec": "0-2",
      "shot_goal": "what this shot accomplishes",
      "visual": "what we see (use visual elements)",
      "camera_motion": "push-in/pan/handheld walk/etc",
      "action": "what moves/changes",
      "subtitle": "short on-screen text",
      "voiceover": "short voiceover",
      "product_focus": "feature or benefit"
    }}
  ]
}}
```
'''

video_generation_prompt = '''Generate a 15-second 16:9 in-feed native video ad based on the detailed script, the product figure, and the start frame.

Rules:
- Must not be static; ensure continuous motion and shot changes.
- Avoid freeze-frame or long still shots.
- Start from the start frame, and keep last ~2 seconds back to the start frame so that the ad can be inserted to the target video.

Now generate the video based on the storyboard JSON below.
The first image is the start frame, and the second image is the product.'''

script_generation_prompt_open = '''
You are a short-form ad director and copywriter.

We need to insert a product placement within the target video. You need to start by designing how to introduce the product within the target video's content, aligning it with user preferences, from the very first frame.

Create a 5-second 16:9 in-feed native ad storyboard.
The ad must feel like organic content and must NOT be static.
The ad must align with user preferences, including the content, style, and elements that users like.
The ad must be strongly relevant to the content of the current video frame.

Inputs:

User Preference:
- content_preference: {content_preference}
- style_preference {style_preference}
- detailed elements {detailed_elements}

Current Video Summary:
- overall_content: {overall_content}
- content_summary: {content_summary}

Product Information:
- product_name: {product_name}
- product_slogan: {product_slogan}
- product_intro: {product_intro}
- product_details: {product_details}

Two reference figures:
- the picture of the product
- the picture of the start frame of the target video

Hard constraints:
1) Must have motion: every scene must include camera motion OR subject motion.
2) Must change at least every 2-3 seconds (new shot, new action, new angle).
3) **Start from the given start frame (the second given image, with description in the scene) and return to the given start frame at the final 2 seconds**
4) Visual personalization must be grounded in provided visual elements (scenes/objects/actions/colors).
5) Product depiction must be faithful; no wrong logos; no shape distortion.
6) We do not need a clear CTA for the product, this is a soft advertisement.
7) Everything in the ad must be in English.

Integration Rule:
- The product must be introduced as a natural object already existing in the target video’s world,
  not as an external promotion.
- Every product appearance must be justified by the ongoing action, environment, or intent
  shown in the target video.
- If the product were removed, the scene should still make logical sense.

Fallback Integration (use only if direct in-world integration is not plausible):
- The target video may be reframed as content playing on a phone or computer screen.
- The camera can pull back into a real-life environment that naturally fits the product,
  then introduce the product through that context.

Output ENGLISH JSON ONLY:
```json
{{
  "duration_sec": 5,
  "format": "16:9",
  "hook": "one sentence hook",
  "scenes": [
    {{
      "sec": "0-2",
      "shot_goal": "what this shot accomplishes",
      "visual": "what we see (use visual elements)",
      "camera_motion": "push-in/pan/handheld walk/etc",
      "action": "what moves/changes",
      "subtitle": "short on-screen text",
      "voiceover": "short voiceover",
      "product_focus": "feature or benefit"
    }}
  ]
}}
```
'''

rewrite_prompt_open = '''Your task is to rewrite the provided input into a single, production-ready video generation prompt.

# Objective
Transform the given materials (storyboard JSON, rules, etc.) into one continuous, purely visual description that can be directly used by a video generation model.

# Core Requirements
- **Visuals Only**: The output must describe *only* what is seen. Strictly exclude all subtitles, captions, voiceovers, dialogue scripts, and on-screen text.
- **Cinematic Flow**: Use fluent, natural English to describe the video as a seamless visual sequence. Avoid choppy sentences or technical lists.
- **Format**: One single paragraph. No markdown, no bullet points, no section headers.
- **Length**: Maximum 400 English words.
- **Exclusion**: Do not mention “storyboard”, “JSON”, or “rewrite task”.

# Content Mapping Rules
- **Translate Context to Visuals**: Convert any narrative intent (from subtitles or scripts) into visual atmosphere, character expressions, lighting, or actions. Do not describe the words themselves.
- **Scene Continuity**: Respect the storyboard's sequence, converting distinct scenes into a smooth visual narrative using transition words.
- **Camera & Action**: Incorporate specific camera motions (pan, zoom, push-in, glide) and subject movements naturally into the sentence structure.
- **Tone**: Maintain the visual style, color palette, and mood defined in the materials.

# Input anchor script
{script}

# Output
Return JSON: {{"rewritten_prompt": "..."}}'''