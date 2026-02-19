# NextAds: Towards Next-generation Personalized Video Advertising

This repository contains the official implementation of our paper on Next-generation Personalized Video Advertising. **NextAds** formulates two representative
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

The pipeline requires the `QILIN` dataset structure. Ensure you configure the dataset paths correctly in the global config section of `main.py`.

```text
qilin_dataset/
├── notes/                     # Parquet files containing note content
├── recommendation_train/      # User interaction history
├── user_feat/                 # User demographics (age, gender)
├── product_library/
│   └── products_raw.json      # Product metadata
└── images/                    # Raw images linked in notes
```

## 🚀 Usage

The main entry point is `main.py`. It supports end-to-end generation, batch submission, and asynchronous polling.

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

For each run, the pipeline generates a comprehensive package inside `ad_runs_v3/user_{uid}/{product_name}/`:
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
* **Input:** `ad_runs` folder and `product_library/products_raw.json`.
* **Output:** `eval_results.csv` (contains scores and detailed reasoning for each dimension).

#### B. Diversity Evaluation 
This script analyzes all videos generated for a single user to ensure the model isn't producing repetitive "template-like" content.

```bash
python eval_diversity.py
```
* **Input:** The CSV generated by the main metrics script.
* **Process:** Uses Gemini's **Inline Data** mode to input multiple videos simultaneously for comparison.

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


