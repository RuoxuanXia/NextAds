import os
import json
import requests
from io import BytesIO
from PIL import Image, UnidentifiedImageError

INPUT_JSON = "product_library/products.json"  
OUTPUT_DIR = "product_library/images"             
OUTPUT_JSON = "product_library/products_output.json"     
TARGET_HEIGHT = 768                               
GAP = 16                                          

def download_image(url: str, timeout: int = 15) -> Image.Image:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"  [Error]  {url}: {e}")
        return None

def stitch_images_horizontally(images: list, target_h: int, gap: int) -> Image.Image:
    if not images:
        return None
    
    resized_images = []
    for img in images:
        w, h = img.size
        scale = target_h / h
        new_w = max(1, int(w * scale))
        resized_images.append(img.resize((new_w, target_h), Image.LANCZOS))
    
    total_width = sum(img.width for img in resized_images) + gap * (len(resized_images) - 1)
    
    canvas = Image.new("RGB", (total_width, target_h), (255, 255, 255))
    
    current_x = 0
    for img in resized_images:
        canvas.paste(img, (current_x, 0))
        current_x += img.width + gap
        
    return canvas

def main():
    print("Download and stitch product images....")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_JSON):
        print(f"❌ Input file not found: {INPUT_JSON}")
        return
        
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        products = json.load(f)
        
    processed_products = []
    
    for i, prod in enumerate(products):
        prod_name = prod.get("product_name", f"product_{i}")
        image_urls = prod.get("image_urls", []) 
   
        if isinstance(image_urls, str):
            image_urls = [image_urls]
            
        print(f"processing [{i+1}/{len(products)}]: {prod_name}")
        
        valid_images = []
        for url in image_urls:
            img = download_image(url)
            if img:
                valid_images.append(img)
                
        if not valid_images:
            print(f"  ⚠️ productt{prod_name} No valid images found, skipping")
            processed_products.append(prod)
            continue

        stitched_img = stitch_images_horizontally(valid_images, TARGET_HEIGHT, GAP)
        
        safe_name = "".join([c if c.isalnum() else "_" for c in prod_name])
        save_path = os.path.join(OUTPUT_DIR, f"{safe_name}.jpg")
        stitched_img.save(save_path, "JPEG", quality=90)
        print(f"  ✅ Successfully stitched and saved to: {save_path}")
        
        prod["product_image"] = save_path
        processed_products.append(prod)
        
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(processed_products, f, ensure_ascii=False, indent=2)
        
    print(f"\n🎉 complete! Pipeline-ready configuration file has been generated: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
