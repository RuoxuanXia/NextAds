# NextAds: Towards Next-generation Personalized Video Advertising

<div align="center">
  <h3>
    👉 <a href="https://nextadsdemo.netlify.app" target="_blank">Click Here to Visit Our Project Page</a> 👈
  </h3>
  
  <a href="https://nextadsdemo.netlify.app" target="_blank">
    <img src="https://img.shields.io/badge/🌐_Project_Page-Live_Demo-blue?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page">
  </a>
</div>


## 📖 Abstract

With the rapid growth of online video consumption, video advertising has become increasingly dominant in the digital advertising landscape. Yet diverse users and viewing contexts makes one-size-fits-all ad creatives insufficient for consistent effectiveness, underlining the importance of personalization. In practice, most personalized video advertising systems follow a retrieval-based paradigm, selecting the optimal one from a small set of professionally pre-produced creatives for each user. Such static and finite inventories limits both the granularity and the timeliness of personalization, and prevents the creatives from being continuously refined based on online user feedback. Recent advances in generative AI make it possible to move beyond retrieval toward optimizing video creatives in a continuous space at serving time.

In this light, we propose **NextAds**, a generation-based paradigm for next-generation personalized video advertising, and conceptualize NextAds with four core components. To enable comparable research progress, we formulate two representative tasks: **personalized creative generation** and **personalized creative integration**, and introduce corresponding lightweight benchmarks. To assess feasibility, we instantiate end-to-end pipelines for both tasks and conduct initial exploratory experiments, demonstrating that GenAI can generate and integrate personalized creatives with encouraging performance. Moreover, we discuss the key challenges and opportunities under this paradigm, aiming to provide actionable insights for both researchers and practitioners and to catalyze progress in personalized video advertising.

<p align="center">
  <img src="pictures/evolution.png" alt="Evolution of Video Advertising" width="60%">
</p>
<p align="center">
  <em><strong>Figure 1:</strong> The evolution of video advertising has largely progressed along the axes of personalization and nativeness, yet high production costs and manual creative bottlenecks prevent the industry from reaching the upper-right corner.</em>
</p>


## 🚀 NextAds Framework 

<p align="center">
  <img src="pictures/intro.png" alt="NextAds Paradigm Shift" width="80%">
</p>
<p align="center">
  <em><strong>Figure 2: Paradigm shift: from retrieval to generation.</strong> NextAds transforms personalized video advertising from static retrieval over a discrete creative inventory to dynamic generation-based optimization in a continuous space.</em>
</p>
---

##  Two Representative Tasks

To enable comparable research progress and assess feasibility, this repository provides end-to-end pipeline instantiations for the two core tasks formulated in **NextAds**:
1. **Pipeline A (Personalized Creative Generation):** Generates personalized ads from scratch by blending user preferences with product characteristics.
2. **Pipeline B (Personalized Creative Integration):** Automatically inserts personalized soft ads into existing target video content.

---

## 🛠️ Global Environment Setup

Ensure you are using Python 3.9+ (tested on macOS and Linux).

1. Clone this repository:
   ```bash
   git clone [https://github.com/anonymous-repo/NextAds.git](https://github.com/anonymous-repo/NextAds.git)
   cd NextAds
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Environment Variables. You will need API keys for the generation models (OpenAI, Sora-2), image hosting (Aliyun OSS), and evaluation models (Gemini):
   ```bash
   export OPENAI_API_KEY="sk-..."
   export OSS_ACCESS_KEY_ID="your_oss_key"
   export SORA2_API_KEY="your_sora_token"
   export GEMINI_API_KEY="your_gemini_api_key"
   ```

## 🎬 Pipeline A: Personalized Creative Generation (PCG)

### 1. Overview
This pipeline consists of the following five key stages:
* **Multimodal User Modeling**: We adopt a dual-stream approach to extract both textual and visual preferences from the user’s interaction history. First, for textual mining, **Director** analyzes click history to identify preference tiers and content style attributes (e.g., tone tags). Simultaneously, for visual extraction, we aggregate user-clicked images and employ a Vision LLM to decode structured visual metadata, capturing abstract “Vibe” descriptors and concrete reusable elements.
* **User-Product Matching**: Next, we employ a bidirectional scoring mechanism to align the extracted user profile with the product information. A VLM evaluates the compatibility between the user’s preference tiers and the product features, selecting the tier that maximizes *Product Match*, *Ad Nativeness*, and *Expressibility*. 
* **Storyboard Generation**: Based on the extracted visual metadata and textual preference, **Director** generates an executable, scene-level storyboard script, specifying camera motions, narration pacing, and visual composition for every 2–3 second slot.
* **Asset Grounding**: To support faithful generation, we implement a **Stitched Reference Strategy** that binds the creative plan to concrete visual assets. The system constructs a *User Visual Collage* from historical images and concatenates it with the official product images, enforcing both user style and product identity.
* **Video Creative Generation**: Finally, the **Producer** executes this plan by feeding the storyboard and the grounded assets into the video generation model to synthesize the final creative.

### 2. Data Preparation (QILIN Dataset)
You can run our pipeline using either the full Parquet dataset or a pre-processed lightweight CSV subset.

**Option 1: Full Dataset (Parquet)**
Ensure you configure the dataset paths correctly in the global config section of `main.py`.
```text
qilin_dataset/
├── notes/                     # Parquet files containing note content
├── recommendation_train/      # User interaction history
├── user_feat/                 # User demographics (age, gender)
├── product_library/
│   ├── products.json          # Product metadata
│   └── images/                # Official product images
└── images/                    # Raw images linked in user notes
```

**Option 2: Pre-processed Subset (CSV)**
For quick testing, we provide `user_subset.csv`, which aggregates user demographics, clicked note texts, and linked image paths.
```csv
user_idx,gender,age,note_idx,note_title,note_content,image_paths
16,female,31-35,1739813,170mL Salon Haircut Comb Hair Dye Bottle Perm Lotion Bottle Bubble Dye Comb,"New arrival, 170mL salon haircut comb, hair dye bottle, perm lotion coloring bottle, bubble hair dye bottle",/path/to/images/3347833.jpg|/path/to/images/3347834.jpg
16,female,31-35,1545593,"Men's side hair grows fast, use a fade comb to cut at home","The most common problem for men is that the side hair grows very fast...",/path/to/images/2887858.jpg
16,female,31-35,896289,Dual-purpose haircut comb,#Haircut[Topic]# #KidsHaircut[Topic]# #GoodThingsRecommendation[Topic]#,
16,female,31-35,1121715,Haircut comb review,"With this comb, can you cut your own hair at home? #Review #Haircut #Comb #Unboxing",
16,female,31-35,886071,Sugar Orange and Gong's Little Orange,"#Phalaenopsis[Topic]# I prefer the color of Sugar Orange, a bright orange, but its growth is particularly poor...",/path/to/images/4535070.jpg|/path/to/images/4535071.jpg
```
*(Note: Multiple image paths are separated by a pipe `|` character. Empty image paths indicate text-only interactions.)*

### 3. Usage
The main entry point is `main.py` (or `Personalized_Creative_Generation.py`). It supports end-to-end generation, batch submission, and asynchronous polling.

**End-to-End Generation:**
```bash
python main.py \
  --users "Userid1, Userid2" \
  --products "ProductA, ProductB" \
  --do_sora \
  --local_out "./output_samples"
```

**High-Concurrency Batch Generation:**
```bash
# Step A: Submit Only
python main.py --users "11094,15067" --products "ALL" --submit_only

# Step B: Poll and Download (Concurrent)
python main.py --users "11094,15067" --poll_only --sora_concurrency 4
```

### 4. Output Structure
Generated inside `ad_runs/user_{uid}/{product_name}/`:
* `storyboard.json`: LLM-generated shot-by-shot script.
* `sora_web_prompt.txt`: Final prompt sent to Sora-2.
* `sora_ref.jpg`: Merged reference image.
* `video.mp4`: Final generated 15s advertisement.
* `review.json`: Metadata linking all assets together.

---

## 🧩 Pipeline B: Personalized Creative Integration (PCI)

### 1. Overview
The PCI Pipeline enables automatic personalized ad insertion. It generates user-preference-aligned soft ads by leveraging target video content, product information, and extracted user interaction history/personalization data.
The PCI Pipeline enables automatic personalized ad insertion through the following key stages:
* **Multimodal User Modeling**: Based on user interaction history (especially engaged videos), we use a VLM to preprocess and annotate preliminary attributes, including topic, textual presentation style, and visual presentation style (e.g., visual tone and camera motion). These are aggregated with video covers to construct a profile capturing the user’s textual, visual, and fine-grained element preferences.
* **Host Video Summarization**: We employ a VLM to generate a structured summary of the target host video, capturing four key aspects: topic, content, visual presentation, and audio characteristics.
* **Integration Decision**: Prioritizing integration smoothness, we bypass the user-product matching stage. Instead, a VLM determines the optimal integration point by identifying the frame in the host video that yields the most natural transition for introducing the product, outputting the chosen point, a brief rationale, and the exact start frame.
* **Storyboard Generation**: Utilizing the user profile, target product, host video summary, and integration decision, the VLM generates a detailed storyboard script. Crucially, the creative is constrained to start and eventually return to the selected integration point, ensuring a seamless visual loop back to the host context.
* **Asset Grounding**: To promote faithful generation and reduce hallucinations, we build a reference visual collage by compositing the target product images, user-preferred elements, and the selected integration start frame. This provides explicit visual grounding for the generation process.
* **Video Creative Integration**: Finally, a video generation model synthesizes the ad creative from the storyboard and grounded assets. The generated segment is then precisely inserted into the host video at the designated integration point.

### 2. Model Configuration
Configure the model settings in the `run.sh` script based on your preference:
* **Close-source Models (e.g., GPT-4o, Sora2):** Edit the `api_key` parameter.
* **Open-source Models (e.g., Qwen3-VL):** We recommend deploying LLMs using `vllm` or `sg-lang`. Edit the `base_url` parameter to point to your deployed model endpoint.

### 3. Data Preparation (MicroLens Dataset)
The anchor dataset for MicroLens is available in the official repository. Please refer to: [westlake-repl/MicroLens](https://github.com/westlake-repl/MicroLens/).

Key parameters:
* `vid`: Video ID in the MicroLens dataset.
* `uid`: User ID specified in `users.json`.

*Edit all necessary file paths (dataset path, output path) in `run.sh` to match your local environment before running.*

**For your convenience, we have uploaded the subset used in our benchmark on [NextAds_PCI](https://1drv.ms/f/c/61ec5ad72cbbbe63/IgBZIy7ntvlETI4N2WHGHZeTASy8CdCLqhHBcco_rEDd55c?e=eRYPwY).**

### 4. Usage
Generate personalized integrated ads with a single command:
```bash
bash run.sh
```

---

## 📊 Evaluation Suite

We provide an automated evaluation pipeline to assess the quality of the generated personalized creatives across five dimensions. All dimensions are scored on a scale of **0-10**. 

*(Note: Metrics marked with **[PCI & PCG]** apply to both tasks, while **[PCI Only]** is specific to PCI.)*

| Dimension | Description | Script |
| :--- | :--- | :--- |
| **Diversity** `[PCI & PCG]` | Evaluates variance in visual style and narrative across different products for the same user. | `eval_diversity.py` |
| **Presentation** `[PCI & PCG]` | Measures how well the video's aesthetic, pacing, and tone align with the user's historical vibe. | `eval_main_metrics.py` |
| **Content** `[PCI & PCG]` | Assesses if the ad highlights product features that resonate with specific user interests. | `eval_main_metrics.py` |
| **Identity Consistency** `[PCI & PCG]` | Verifies the factual accuracy of the product's appearance, logo, and selling points. | `eval_main_metrics.py` |
| **Integration** `[PCI Only]` | Evaluates how naturally the advertisement is woven into the video content, considering contextual flow, thematic consistency, and user experience disruption. | `eval_main_metrics.py` |

### 1. How to Run

**A. Multi-Metric Evaluation (Main Results)**
Performs a three-way evaluation (Presentation, Content, Identity) by comparing videos against user historical notes and official product metadata.
```bash
python eval_main_metrics.py
```

**B. Diversity Evaluation**
Analyzes all videos generated for a single user to ensure the model avoids repetitive "template-like" content.
```bash
python eval_diversity.py
```

### 2. Output Structure
Evaluation results are saved in a structured CSV format, including both quantitative scores and qualitative reasoning:
```csv
user_id, product_name, presentation_score, presentation_reason, content_score, content_reason, ...
```




