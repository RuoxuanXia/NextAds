# NextAds: Towards Next-generation Personalized Video Advertising

<div align="center">

<a href="https://nextadsdemo.netlify.app">
<img src="https://img.shields.io/badge/🌐_CLICK_HERE_TO_VIEW-INTERACTIVE_WEB_DEMO-bd00ff?style=for-the-badge&logoColor=00f2ff&labelColor=1a0529&color=00f2ff" alt="Interactive Web Demo" height="50">
</a>

<br><br>

<p>
  <b><i>Comprehensive Gallery for <span style="color: #bd00ff;">Personalized Creative Generation</span></i></b>
  <br>
  <b><i>and <span style="color: #00f2ff;">Personalized Creative Integration</span> Showcases</i></b>
</p>

> **Note to Reviewers:** This website has been fully anonymized for double-blind review purposes. It contains the comprehensive qualitative results and interactive video galleries mentioned in the paper.

---
</div>


This repository contains the official implementation of our paper on **NextAds: Towards Next-generation Personalized Video Advertising**.
**NextAds** formulates two representative
tasks: personalized creative generation and personalized creative
integration, and introduce corresponding lightweight benchmarks.
To assess feasibility, we instantiate end-to-end pipelines for both
tasks and conduct initial exploratory experiments, demonstrating
that GenAI can generate and integrate personalized creatives with
encouraging performance. Moreover, we discuss the key challenges
and opportunities under this paradigm, aiming to provide actionable
insights for both researchers and practitioners and to catalyze
progress in personalized video advertising.

## 📝 Overview

The pipeline of Personalized Creative Generation consists of the following key stages:
* **Multimodal User Modeling**: We adopt a dual-stream approach to extract both textual and visual preferences from the user’s interaction history. First, for textual mining, **Director** analyzes click history to identify preference tiers and content style attributes (e.g., tone tags). Simultaneously, for visual extraction, we aggregate user-clicked images and employ a Vision LLM to decode structured visual metadata, capturing abstract “Vibe” descriptors and concrete reusable elements.
* **User-Product Matching**: Next, we employ a bidirectional scoring mechanism to align the extracted user profile with the product information. A VLM evaluates the compatibility between the user’s preference tiers and the product features, selecting the tier that maximizes *Product Match*, *Ad Nativeness*, and *Expressibility*. This process prioritizes scenarios that naturally bridge user interests with commercial intent.
* **Storyboard Generation**: Based on the extracted visual metadata and textual preference, **Director** generates an executable, scene-level storyboard script, which specifies camera motions, narration pacing, and visual composition for every 2–3 second slot.
* **Asset Grounding**: To support faithful generation, we implement a **Stitched Reference Strategy** that binds the creative plan to concrete visual assets. The system constructs a *User Visual Collage* from historical images and concatenates it with the official product images. This composite image acts as a strict grounding constraint, enforcing both the user’s visual style (left-side reference) and the product’s identity (right-side reference).
* **Video Creative Generation**: Finally, the **Producer** executes this plan by feeding the storyboard and the grounded assets into the video generation model, synthesizing the final creative, blending the user’s preferred aesthetics with the product’s key features.

## 🛠️ Environment Setup

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

3. Set up Environment Variables. You will need API keys for OpenAI (GPT-4o), Aliyun OSS (for image hosting), and Sora-2 :
   ```bash
   export OPENAI_API_KEY="sk-..."
   export OSS_ACCESS_KEY_ID="your_oss_key"
   export SORA2_API_KEY="your_sora_token"
   ```

## 📂 Data Preparation

You can run our pipeline using either the full Parquet dataset or a pre-processed lightweight CSV subset.

### Option 1: Full Dataset (Parquet)
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

### Option 2: Pre-processed Subset (CSV)
For quick testing and easier reproduction, we also provide a pre-processed user subset. This CSV file aggregates user demographics, clicked note texts, and linked image paths into a single flat structure.

**Format Example (`user_subset.csv`):**
```csv
user_idx,gender,age,note_idx,note_title,note_content,image_paths
16,female,31-35,1739813,170mL salon hair comb hair dye bottle perm potion bottle bubble dye comb,"New arrivals, 170mL salon hair comb, hair dye bottle, perm potion coloring bottle, bubble dye bottle",/path/to/images/3347833.jpg|/path/to/images/3347834.jpg
16,female,31-35,1545593,"Men's hair grows fast on the sides, use a fade comb to cut it at home","The most common problem for men is that the sides grow out very quickly... With a fade comb, no skills are needed, you can cut it yourself at home...",/path/to/images/2887858.jpg
16,female,31-35,896289,Dual-purpose haircut comb,#Haircut[Topic]# #KidsHaircut[Topic]# #GoodThingsToRecommend[Topic]#,
16,female,31-35,1121715,Haircut comb review,"With this comb, you can cut your hair at home? #Review #Haircut #Comb",
16,female,31-35,886071,Shatangju and little mandarin orchids,"#Phalaenopsis[Topic]# I prefer the color of Shatangju, a bright orange, but it doesn't grow well...",/path/to/images/4535070.jpg|/path/to/images/4535071.jpg
```
*(Note: Multiple image paths are separated by a pipe `|` character. Empty image paths indicate text-only interactions. To save space, long text contents in this example are truncated.)*

## 🚀 Usage

The main entry point is `Personalized_Creative_Generation.py`. It supports end-to-end generation, batch submission, and asynchronous polling.

### 1. End-to-End Generation
To generate personalized videos for a specific user and specific products:

```bash
python main.py \
  --users "Userid1 , Userid2 ,..." \
  --products "ProductA , ProductB ,..." \
  --do_sora \
  --local_out "./output_samples"
```

### 2. High-Concurrency Batch Generation
If you are generating videos for a large number of users, you can decouple the submission and downloading processes to prevent timeouts:

**Step A: Submit Only**
```bash
python main.py --users "11094,15067,576" --products "ALL" --submit_only
```

**Step B: Poll and Download (Concurrent)**
```bash
python main.py --users "11094,15067,576" --poll_only --sora_concurrency 4
```
## 📁 Output Structure

For each run, the pipeline generates a comprehensive package inside `ad_runs/user_{uid}/{product_name}/`:
* `storyboard.json`: The LLM-generated shot-by-shot script.
* `sora_web_prompt.txt`: The final prompt sent to Sora-2.
* `sora_ref.jpg`: The merged reference image (User Preference Grid + Product Image).
* `video.mp4`: The final generated 15s advertisement.
* `review.json`: Metadata linking all assets together.

### Evaluation Metrics

| Dimension | Description | Script |
| :--- | :--- | :--- |
| **Diversity** | Evaluates the variance in visual style and narrative across different products for the same user. | `eval_diversity.py` |
| **Presentation** | Measures how well the video's aesthetic, pacing, and tone align with the user's historical vibe. | `eval_main_metrics.py` |
| **Content** | Assesses if the ad highlights product features that resonate with specific user interests. | `eval_main_metrics.py` |
| **Identity Consistency** | Verifies the factual accuracy of the product's appearance, logo, and selling points. | `eval_main_metrics.py` |

### 1. How to Run

Before running, ensure you have set your Gemini API key and proxy (if necessary):
```bash
export GEMINI_API_KEY="your_gemini_api_key"
```

#### A. Multi-Metric Evaluation (Main Results)
This script performs a three-way evaluation (Presentation, Content, Identity) for each video by comparing it against user historical notes and official product metadata.

```bash
python eval_main_metrics.py
```

#### B. Diversity Evaluation 
This script analyzes all videos generated for a single user to ensure the model isn't producing repetitive "template-like" content.

```bash
python eval_diversity.py
```

### 2. Scoring Criteria
All dimensions are scored on a scale of **0-10**:
* **10:** Perfect alignment / No noticeable overlap (for Diversity).
* **7-9:** Good alignment with minor variations.
* **4-6:** Moderate alignment / Partial consistency.
* **1-3:** Poor alignment / Significant factual errors.
* **0:** Completely irrelevant / Failed generation.

## 📁 Output Structure

The evaluation results are saved in a structured CSV format, including both quantitative scores and qualitative reasoning:
```csv
user_id, product_name, presentation_score, presentation_reason, content_score, content_reason, ...
```


