import os
import json
import argparse
from prompts import (
    video_summary_prompt, 
    preference_summary_prompt,
    insertion_frame_prompt, 
    script_generation_prompt, 
    video_generation_prompt
)
from utils import (
    load_json, 
    save_json,
    parse_json_output, 
    encode_image, 
    process_frames, 
    process_images_using_llm, 
    submit_sora, 
    retrieve_video, 
    get_fps,
    save_base64_frame_to_jpg
)

def parse_args():
    parser = argparse.ArgumentParser(description="Personalized advertise generation based on user's video history")

    # Data path
    parser.add_argument("--user_persona_path", type=str, required=True, help="Path to the user persona JSON file")
    parser.add_argument("--product_path", type=str, required=True, help="Path to the product JSON file")
    parser.add_argument("--save_path", type=str, required=True, help="Path to the save dir")

    parser.add_argument("--video_dir", type=str, required=True, help="Path to videos")
    parser.add_argument("--covers_dir", type=str, required=True, help="Path to covers")

    # Model settings
    parser.add_argument("--llm_api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--llm_base_url", type=str, default=None, help="OpenAI base URL")
    parser.add_argument("--llm_model_name", type=str, default="gpt-4o")
    parser.add_argument("--sora_api_key", type=str, required=True)

    return parser.parse_args()

def agent(args):
    # Step0: Data preparation
    user_persona = load_json(args.user_persona_path)
    products = load_json(args.product_path)

    for user_id, user in enumerate(user_persona):
        print("*" * 15 + f"Processing No.{user_id} user, with topic {user['preference']['topic']}" + "*" * 15)
        target_vid = user['vids'][-1]
        interacted_vids = user['vids'][:-1]
        preference = user['preference']
        user_save_dir = os.path.join(args.save_path, f"user_{user_id}")
        os.makedirs(user_save_dir, exist_ok=True)
        try:
            # Step1: Summarize the target video
            if not os.path.exists(os.path.join(user_save_dir, "target_video_summary.json")):
                print(f"Summarizing the target video...")
                target_video_fps = get_fps(os.path.join(args.video_dir, f"{target_vid}.mp4"))
                target_video_frames = process_frames(os.path.join(args.video_dir, f"{target_vid}.mp4"), target_video_fps * 30)
                target_video_summary = parse_json_output(process_images_using_llm(api_key=args.llm_api_key, 
                    base_url=args.llm_base_url, figs=target_video_frames[::target_video_fps], 
                    prompt=video_summary_prompt, model_name=args.llm_model_name))
                assert target_video_summary is not None
                save_json(target_video_summary, os.path.join(user_save_dir, "target_video_summary.json"))
                print(f'Finished target video summary, check {os.path.join(user_save_dir, "target_video_summary.json")}')
            else:
                target_video_summary = load_json(os.path.join(user_save_dir, "target_video_summary.json"))

            # Step2: User persona extraction
            if not os.path.exists(os.path.join(user_save_dir, "preference_summary.json")):
                print(f"Summarizing user's persona...")
                interacted_covers = [encode_image(os.path.join(args.covers_dir, f"{vid}.jpg")) for vid in interacted_vids]
                preference_summary = parse_json_output(process_images_using_llm(api_key=args.llm_api_key, 
                    base_url=args.llm_base_url, figs=interacted_covers, 
                    prompt=preference_summary_prompt.format(topic=preference['topic'], tone=preference['presentation']['tone'], camera_work=preference['presentation']['camera_work'], narrative_template=preference['narrative_template']), 
                    model_name=args.llm_model_name))
                assert preference_summary is not None
                save_json(preference_summary, os.path.join(user_save_dir, "preference_summary.json"))
                print(f'Finished user persona summary, check {os.path.join(user_save_dir, "preference_summary.json")}')
            else:
                preference_summary = load_json(os.path.join(user_save_dir, "preference_summary.json"))
        except Exception as e:
            print(f"Fail on user{user_id}: {e}")
            continue

        # Load all products
        products_info = load_json(args.product_path)

        for pid in user['products']:
            product = products_info[pid]
            print(f"Working on product: {product['product_name']}")
            product_save_dir = os.path.join(user_save_dir, product['product_name'].replace(' ', '_'))
            os.makedirs(product_save_dir, exist_ok=True)

            try:
                # Step3: Decide the insertion timestamp
                if not os.path.exists(os.path.join(product_save_dir, "insertion.json")):
                    print(f"Deciding which timestamp to insert the ad...")
                    insertion_prompt = insertion_frame_prompt.format(product_name=product['product_name'], product_details=product['product_details'])
                    target_video_fps = get_fps(os.path.join(args.video_dir, f"{target_vid}.mp4"))
                    # print(f"FPS: {target_video_fps}")
                    target_video_frames = process_frames(os.path.join(args.video_dir, f"{target_vid}.mp4"), max_frames=30 * target_video_fps)
                    # print(f"Total frames: {len(target_video_frames)}")
                    insertion_details = parse_json_output(process_images_using_llm(api_key=args.llm_api_key, 
                        base_url=args.llm_base_url, figs=target_video_frames[::target_video_fps], 
                        prompt=insertion_prompt, model_name="gpt-4.1-mini"))
                    assert insertion_details is not None
                    save_json(insertion_details, os.path.join(product_save_dir, "insertion.json"))
                    print(f"Insert on the {insertion_details['insertion_timestamp'] - 1}s")
                    start_frame = target_video_frames[(insertion_details['insertion_timestamp'] - 1) * target_video_fps]
                    save_base64_frame_to_jpg(start_frame, os.path.join('/NAS/xuyiy/PersonaAd/frames', f'user{user_id}_{product["product_name"].replace(" ", "_")}.jpg'))
                # else:
                #     insertion_details = load_json(os.path.join(product_save_dir, "insertion.json"))

                if not os.path.exists(os.path.join(product_save_dir, "script.json")):
                    print(f"Generating script...")
                    # Step4: Script generation
                    script_prompt = script_generation_prompt.format(content_preference=preference_summary['content_preference'], style_preference=preference_summary['style_preference'], detailed_elements=preference_summary['detailed_elements'], 
                        overall_content=target_video_summary['content_summary'], content_summary=target_video_summary['content_summary'], 
                        product_name=product['product_name'], product_slogan=product['product_slogan'], product_intro=product['product_intro'], product_details=product['product_details'])

                    product_img = [encode_image(product['product_image']), encode_image(os.path.join('/NAS/xuyiy/PersonaAd/frames', f'user{user_id}_{product["product_name"].replace(" ", "_")}.jpg'))]
                    script = parse_json_output(process_images_using_llm(api_key=args.llm_api_key, 
                        base_url=args.llm_base_url, figs=product_img, 
                        prompt=script_prompt, model_name=args.llm_model_name))
                    assert script is not None
                    save_json(script, os.path.join(product_save_dir, "script.json"))
                    print(f'Finished script generation, check {os.path.join(product_save_dir, "script.json")}')
                else:
                    script = load_json(os.path.join(product_save_dir, "script.json"))

                if not os.path.exists(os.path.join(product_save_dir, "result.mp4")):
                    print(f"Generating video...")
                    # Step5: Advertise generation
                    gen_prompt = video_generation_prompt + json.dumps(script, indent=2)

                    task_id = submit_sora(api_key=args.sora_api_key, prompt=gen_prompt, 
                          image_urls=[product['product_url'], product['startframe_url']])
                    retrieve_video(task_id, os.path.join(product_save_dir, "result.mp4"), args.sora_api_key)
                    print(f'Finished video generation, check {os.path.join(product_save_dir, "result.mp4")}')
            except Exception as e:
                print(f"Fail to generate ad for {product['product_name']}: {e}")

if __name__ == "__main__":
    args = parse_args()
    agent(args)
