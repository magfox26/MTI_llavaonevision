import torch
import json
import os
import gc
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import requests
import random

from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration


def load_image(image_file):
    """Load image from URL or local file"""
    if image_file.startswith("http"):
        response = requests.get(image_file, stream=True)
        image = Image.open(response.raw).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files, image_folder):
    """Load multiple images and return them"""
    base_folder = Path(image_folder)
    images = []
    for image_file in image_files:
        image_path = base_folder / image_file.strip()
        if image_path.exists():
            images.append(load_image(str(image_path)))
        else:
            print(f"Image not found: {image_path}")
    return images


def generate(text, image_files, image_folder, tgt_lang="English"):
    """Generate translation for the given image(s)"""
    if isinstance(image_files, list) and len(image_files) > 0:
        if image_folder:
            # Load multiple images from a folder
            raw_image = load_images(image_files, image_folder)[0]  # Taking first image
        else:
            # Direct path provided
            raw_image = load_image(image_files[0])
    else:
        return "No image provided"

    # Define system and user prompts
    system_prompt = """Please strictly follow the steps below to process the text in the image:
1. **Comprehensive Recognition**: Extract all visible text elements in the image (including words, numbers, symbols, special characters)
2. **Translatable text**: Accurate translation into target language
3. **Format retention**:
   - Maintain original text alignment
   - Original line breaks and paragraph structure are preserved
4. **Quality check**:
   (1) Verify that all text blocks have been processed
   (2) Verify terminology accuracy
**Output Standardization**:
1. prohibit inclusion of original text
2. Prohibit the addition of explanatory notes
3. output only the final translation result
4. Only output the translated text in the target language
"""

    # Define conversation with the new prompt template
    conversation = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Please translate the text in the image into {tgt_lang}. {text}"}
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Process inputs
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(device, torch.float16)

    # Generate output
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0
        )

    # Get the raw generated text
    result = processor.decode(output[0][2:], skip_special_tokens=True)

    # Extract only the translation part
    # Look for the assistant's response after the prompt
    if "assistant" in result:
        # Extract only the text after "assistant"
        translation = result.split("assistant", 1)[1].strip()
    else:
        # Fallback if the format is different
        translation = result.strip()

    # Clean up memory
    del inputs, output
    torch.cuda.empty_cache()
    gc.collect()

    return translation


def eval_ocrmt():
    """Evaluate OCRMT task"""
    output_path = f"/mnt/data/users/liamding/data/liu_SFT/MTI/evaluations/ocrmt/ocr_mt/"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    results = {}
    image_folder = "/mnt/data/users/liamding/data/dataset/OCRMT30K-refine/whole_image_v2/"
    img_source = json.load(
        open("/mnt/data/users/liamding/data/dataset/OCRMT30K-refine/original_data/original_test_1000.json", "r",
             encoding="utf-8"))

    for img, item in tqdm(img_source.items()):
        # Generate model output
        text = ""
        ocr_outputs = generate(text, [img.strip()], image_folder, tgt_lang="English")

        # Process text consistently
        def process_text(text_list):
            parts = []
            for text in text_list:
                split_parts = [p.strip() for p in text.split(",") if p.strip()]
                parts.extend(split_parts)
            return " ".join(parts)

        ref_processed = process_text(item["en"])
        src_processed = process_text(item["zh"])

        results[img] = {
            "ref": ref_processed,
            "src": src_processed,
            "mt": ocr_outputs
        }

    json.dump(results, open(f"{output_path}/original-new.json", "w"), ensure_ascii=False, indent=4)


def eval_mtit6(lang="en2zh"):
    """Evaluate MTIT6 task"""
    config = {
        "en2zh": ("/mnt/data/users/liamding/data/dataset/AnyTrans-refine/images/en2zh/", "Chinese"),
        "zh2en": ("/mnt/data/users/liamding/data/dataset/AnyTrans-refine/images/zh2en/", "English"),
        "ko2zh": ("/mnt/data/users/liamding/data/dataset/AnyTrans-refine/images/ko2zh/", "Chinese"),
        "zh2ko": ("/mnt/data/users/liamding/data/dataset/AnyTrans-refine/images/zh2ko/", "Korean"),
        "ja2zh": ("/mnt/data/users/liamding/data/dataset/AnyTrans-refine/images/ja2zh/", "Chinese"),
        "zh2ja": ("/mnt/data/users/liamding/data/dataset/AnyTrans-refine/images/zh2ja/", "Japanese")
    }

    image_folder, tgt_lang = config[lang]

    # Reference file suffixes
    ref_suffix = {
        'en2zh': '231',
        'zh2en': '191',
        'ko2zh': '196',
        'zh2ko': '170',
        'ja2zh': '211',
        'zh2ja': '200'
    }

    output_path = f"/mnt/data/users/liamding/data/liu_SFT/MTI/evaluations/mtit6/{lang}/ocr_mt"
    ref_path = f"/mnt/data/users/liamding/data/dataset/AnyTrans-refine/{lang}_{ref_suffix[lang]}.json"

    Path(output_path).mkdir(parents=True, exist_ok=True)
    ref_data = json.load(open(ref_path, "r", encoding="utf-8"))
    results = {}

    for img, item in tqdm(ref_data.items()):
        # Split source and target language
        src_lang, tgt_lang_code = lang.split('2')
        src_text = item[src_lang]
        ref_text = item[tgt_lang_code]

        # Process comma-separated fields
        def process_text(text_list):
            parts = []
            for text in text_list:
                split_parts = [p.strip() for p in text.split(",") if p.strip()]
                parts.extend(split_parts)
            return " ".join(parts)

        src_processed = process_text(src_text)
        ref_processed = process_text(ref_text)

        # Generate model output
        mt_output = generate("", [img], image_folder, tgt_lang)

        results[img] = {
            "ref": ref_processed,
            "src": src_processed,
            "mt": mt_output
        }

    json.dump(results, open(f"{output_path}/original-new.json", "w"), ensure_ascii=False, indent=4)


def eval_mit10(lang="en2zh"):
    """Evaluate MIT10 task"""
    # Split source and target language
    src_lang, tgt_lang_code = lang.split('2')

    # Language name mapping
    lang_name_map = {
        "en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
        "de": "German", "pt": "Portuguese", "es": "Spanish", "fr": "French",
        "it": "Italian", "ru": "Russian", "ar": "Arabic", "th": "Thai",
        "tr": "Turkish", "hi": "Hindi"
    }

    tgt_lang = lang_name_map[tgt_lang_code]

    # Base directory
    base_folder = "/mnt/data/users/liamding/data/dataset/MIT-10M/data/small"

    # Source JSON file path
    img_source_path = f"/mnt/data/users/liamding/data/dataset/MIT-10M/test/test_{src_lang}.json"

    # Output directory path
    output_path = f"/mnt/data/users/liamding/data/liu_SFT/MTI/evaluations/mit10/{lang}/ocr_mt/"
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Read source JSON file
    with open(img_source_path, "r", encoding="utf-8") as f:
        img_source = json.load(f)

    results = {}

    # Process JSON format
    for img_key, translations in tqdm(img_source.items(), desc=f"Evaluating {lang}"):
        # Process image filename and build correct path
        img_filename = os.path.basename(img_key)

        # Try multiple possible path formats
        possible_paths = []

        # 1. Direct use of img_key (might be full path)
        possible_paths.append(os.path.join(base_folder, img_key))

        # 2. Use first two characters as subdirectory
        if '_' in img_filename:
            img_id = img_filename.split('_')[1].split('.')[0]
            if len(img_id) >= 4:
                subdir1 = img_id[:2]
                subdir2 = img_id[2:4]
                possible_paths.append(os.path.join(base_folder, src_lang, subdir1, subdir2, img_filename))

        # 3. Use standard format (omit language prefix)
        if '_' in img_filename:
            img_base = img_filename.split('_')[1]
            img_id = img_base.split('.')[0]
            if len(img_id) >= 4:
                subdir1 = img_id[:2]
                subdir2 = img_id[2:4]
                possible_paths.append(os.path.join(base_folder, src_lang, subdir1, subdir2, img_base))

        # Find first existing path
        img_path = None
        for path in possible_paths:
            if os.path.exists(path):
                img_path = path
                break

        # Skip if all paths don't exist
        if img_path is None:
            print(f"Cannot find image for {img_key}. Tried paths: {possible_paths}")
            continue

        # Process source and reference text
        src_text = translations[src_lang]
        ref_text = translations[tgt_lang_code]

        # Clean and join text
        src_text_joined = "".join(src_text).replace("\n", " ").strip() if isinstance(src_text,
                                                                                     list) else src_text.replace("\n",
                                                                                                                 " ").strip()
        ref_text_joined = "".join(ref_text).replace("\n", " ").strip() if isinstance(ref_text,
                                                                                     list) else ref_text.replace("\n",
                                                                                                                 " ").strip()

        # Generate model output
        try:
            ocr_outputs = generate("", [img_path], "", tgt_lang=tgt_lang)

            results[img_key] = {
                "ref": ref_text_joined,
                "src": src_text_joined,
                "mt": ocr_outputs,
            }
        except Exception as e:
            print(f"Error processing {img_key}: {e}")
            continue

    # Save results
    json.dump(results,
              open(f"{output_path}/original-new.json", "w"),
              ensure_ascii=False,
              indent=4)

    print(f"Completed {lang}. Results saved to {output_path}/original-new.json")
    print(f"Successfully processed images: {len(results)}")


def eval_mit10_all(src_lang="en"):
    """Evaluate all MIT10 language combinations for a source language"""
    # Language code list (14 languages)
    all_langs = ["en", "zh", "ja", "ko", "de", "pt", "es", "fr", "it", "ru", "ar", "th", "tr", "hi"]

    # Full language name mapping
    lang_name_map = {
        "en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
        "de": "German", "pt": "Portuguese", "es": "Spanish", "fr": "French",
        "it": "Italian", "ru": "Russian", "ar": "Arabic", "th": "Thai",
        "tr": "Turkish", "hi": "Hindi"
    }

    # Translate from source language to all other languages
    for tgt_lang in all_langs:
        # Skip translation from source to same source
        if tgt_lang == src_lang:
            continue

        # Build language pair (e.g., 'en2zh')
        lang_pair = f"{src_lang}2{tgt_lang}"
        print(f"Starting {lang_pair} translation task...")

        # Call single language pair evaluation function
        eval_mit10(lang_pair)


def shuffle_without_fixed_positions(img_source):
    """Randomly shuffle image order ensuring no element stays in original position"""
    indices = list(range(len(img_source)))
    while True:
        random.shuffle(indices)
        # Check if all elements have moved from original positions
        if all(i != indices[i] for i in range(len(indices))):
            break
    shuffled_imgs = [img_source[i] for i in indices]
    return shuffled_imgs


# Main execution
if __name__ == "__main__":
    # Model configuration
    model_id = "/mnt/data/users/liamding/data/liu_SFT/outcome_mit10m_sample500/v0-20250318-023742/checkpoint-4875-merged"
    device = "cuda"

    # Load model and processor
    print(f"Loading model {model_id}...")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    model.eval()

    # Uncomment the evaluation tasks you want to run

    # MIT10 evaluation - evaluate all language combinations
    # Evaluate English to all other languages
    #eval_mit10_all("en")  # Evaluates en2zh, en2ja, en2ko, ...

    eval_mit10("en2tr")
    eval_mit10("en2th")
    eval_mit10("en2hi")
    eval_mit10("en2ar")
    # Evaluate Chinese to all other languages
    eval_mit10_all("zh")  # Evaluates zh2en, zh2ja, zh2ko, ...

    # Evaluate other source languages
    eval_mit10_all("de")  # German to all other languages
    eval_mit10_all("es")  # Spanish to all other languages
    eval_mit10_all("fr")  # French to all other languages
    #eval_mit10_all("it")  # Italian to all other languages
    #eval_mit10_all("ja")  # Japanese to all other languages
    #eval_mit10_all("pt")  # Portuguese to all other languages

    # To evaluate specific language pairs, uncomment:
    #eval_mit10("de2ar")

    # Other evaluation tasks
    #eval_ocrmt()
    #eval_mtit6(lang="en2zh")