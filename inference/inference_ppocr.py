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


def load_ocr_results(json_file, image_key):
    """Load OCR results for a specific image from JSON file"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)

        # Try different key formats for matching
        if image_key in ocr_data:
            ocr_result = ocr_data[image_key]
        elif image_key.split('/')[-1] in ocr_data:
            # Try with just the filename
            ocr_result = ocr_data[image_key.split('/')[-1]]
        else:
            # Try to find a key that ends with our image_key
            matching_keys = [k for k in ocr_data.keys() if k.endswith(image_key) or image_key.endswith(k)]
            if matching_keys:
                ocr_result = ocr_data[matching_keys[0]]
            else:
                print(f"No matching OCR key found for {image_key}")
                return ""

        # Get OCR output and convert to string with newline separators
        ocr_output = ocr_result.get("output", []) if isinstance(ocr_result, dict) else []
        ocr_text = "\n".join(ocr_output) if isinstance(ocr_output, list) else str(ocr_output)

        return ocr_text
    except Exception as e:
        print(f"Error loading OCR results for {image_key} from {json_file}: {e}")
        return ""


def generate(text, image_files, image_folder, tgt_lang="English", ocr_json_path=None):
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

    # Load OCR results if JSON path is provided
    ocr_text = ""
    if ocr_json_path and image_files:
        image_key = image_files[0]
        ocr_text = load_ocr_results(ocr_json_path, image_key)

    # Define system instructions
    system_instructions = """Please strictly follow the steps below to process the text in the image:
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

    # Add OCR text to the instructions if available
    ocr_instruction_part = ""
    if ocr_text and ocr_text.strip():
        ocr_instruction_part = f"\n\n[OCR_TEXT_FOR_MODEL_REFERENCE]\n{ocr_text}"
        print("OCR text included in prompt")
    else:
        print("No OCR text available to include")

    # Create the user message with all instructions included
    user_message = f"""System Instructions: {system_instructions}

[OCR_TEXT_FOR_MODEL_REFERENCE]
{ocr_text}

(Please do not include the above original text in the final output, just the translation!)

Please translate the text in the image into {tgt_lang}. {text}"""

    # Skip the chat template and directly construct the prompt in the format shown in debug output
    prompt = f"<|im_start|>user <image>\n{user_message} <|im_end|><|im_start|>assistant"

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


def get_ocr_path(task, lang_pair=None):
    """Get OCR file path based on task and language pair"""
    if task == "ocrmt":
        return os.path.join(ocr_root, "ppocr_ocrmt.json")
    elif task == "mtit6":
        return os.path.join(ocr_root, f"ppocr_{lang_pair}.json")
    elif task == "mit10":
        src_lang = lang_pair.split('2')[0] if lang_pair else None
        return os.path.join(ocr_root, f"ppocr_mit10_{src_lang}.json")
    else:
        print(f"Unknown task: {task}")
        return None


def eval_ocrmt():
    """Evaluate OCRMT task"""
    final_output_path = f"{output_path}/ocrmt/given_ocr_VL/"
    Path(final_output_path).mkdir(parents=True, exist_ok=True)
    results = {}
    image_folder = f"{root}/OCRMT30K-refine/whole_image_v2/"
    img_source = json.load(
        open(f"{root}/OCRMT30K-refine/original_data/original_test_1000.json", "r", encoding="utf-8"))

    # Get OCR path for OCRMT task
    ocr_json_path = get_ocr_path("ocrmt")

    for img, item in tqdm(img_source.items()):
        text = ""
        ocr_outputs = generate(text, [img.strip()], image_folder, tgt_lang="English", ocr_json_path=ocr_json_path)

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

    json.dump(results, open(f"{final_output_path}/{output_name}", "w"), ensure_ascii=False, indent=4)


def eval_mtit6(lang="en2zh"):
    """Evaluate MTIT6 task"""
    config = {
        "en2zh": (f"{root}/AnyTrans-refine/images/en2zh/", "Chinese"),
        "zh2en": (f"{root}/AnyTrans-refine/images/zh2en/", "English"),
        "ko2zh": (f"{root}/AnyTrans-refine/images/ko2zh/", "Chinese"),
        "zh2ko": (f"{root}/AnyTrans-refine/images/zh2ko/", "Korean"),
        "ja2zh": (f"{root}/AnyTrans-refine/images/ja2zh/", "Chinese"),
        "zh2ja": (f"{root}/AnyTrans-refine/images/zh2ja/", "Japanese")
    }

    image_folder, tgt_lang = config[lang]

    # Get OCR path for MTIT6 task with specific language pair
    ocr_json_path = get_ocr_path("mtit6", lang)

    # Reference file suffixes
    ref_suffix = {
        'en2zh': '231',
        'zh2en': '191',
        'ko2zh': '196',
        'zh2ko': '170',
        'ja2zh': '211',
        'zh2ja': '200'
    }

    final_output_path = f"{output_path}/mtit6/{lang}/given_ocr_VL/"
    ref_path = f"{root}/AnyTrans-refine/{lang}_{ref_suffix[lang]}.json"

    Path(final_output_path).mkdir(parents=True, exist_ok=True)
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

        # Generate model output with OCR results
        mt_output = generate("", [img], image_folder, tgt_lang, ocr_json_path=ocr_json_path)

        results[img] = {
            "ref": ref_processed,
            "src": src_processed,
            "mt": mt_output
        }

    json.dump(results, open(f"{final_output_path}/{output_name}", "w"), ensure_ascii=False, indent=4)


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

    # Get OCR path for MIT10 task with specific source language
    ocr_json_path = get_ocr_path("mit10", lang)

    # Verify OCR file exists
    if not os.path.exists(ocr_json_path):
        print(f"WARNING: OCR file not found at {ocr_json_path}")
    else:
        print(f"Found OCR file: {ocr_json_path}")

    # Base directory
    base_folder = f"{root}/MIT-10M/data/small"

    # Source JSON file path
    img_source_path = f"{root}/MIT-10M/test/test_{src_lang}.json"

    # Output directory path
    final_output_path = f"{output_path}/mit10/{lang}/given_ocr_VL/"
    Path(final_output_path).mkdir(parents=True, exist_ok=True)

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

        # Direct use of img_key (might be full path)
        possible_paths.append(os.path.join(base_folder, img_key))

        # Use first two characters as subdirectory
        if '_' in img_filename:
            img_id = img_filename.split('_')[1].split('.')[0]
            if len(img_id) >= 4:
                subdir1 = img_id[:2]
                subdir2 = img_id[2:4]
                possible_paths.append(os.path.join(base_folder, src_lang, subdir1, subdir2, img_filename))

        # Use standard format (omit language prefix)
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

        # Generate model output with OCR results
        try:
            # Try with both absolute path and relative path for OCR lookup
            ocr_outputs = generate("", [img_path], "", tgt_lang=tgt_lang, ocr_json_path=ocr_json_path)

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
              open(f"{final_output_path}/{output_name}", "w"),
              ensure_ascii=False,
              indent=4)

    print(f"Completed {lang}. Results saved to {final_output_path}/{output_name}")
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


# For testing OCR file loading directly
def test_ocr_loading(src_lang="ja"):
    """Test OCR loading for a specific source language"""
    ocr_json_path = get_ocr_path("mit10", f"{src_lang}2zh")  # Any target language will work as we only need src_lang

    try:
        with open(ocr_json_path, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
            print(f"Successfully loaded OCR file with {len(ocr_data)} entries")
            # Print a sample entry
            sample_key = list(ocr_data.keys())[0]
            print(f"Sample entry for key '{sample_key}':")
            print(ocr_data[sample_key])
    except Exception as e:
        print(f"Error loading OCR file {ocr_json_path}: {e}")


if __name__ == "__main__":
    # User-configurable settings
    model_id = "" # Model path,like "output/v0-20250318-023742/checkpoint-4875"
    root = ""  # Base directory for all datasets,like "/mnt/data/users/liamding/data/dataset"
    output_path = ""  # Base directory for output,like "/mnt/data/users/liamding/data/liu_SFT/MTI/evaluations"
    output_name = ""  # Output filename，like original.json
    ocr_root = ""  # PPOCR root directory，like "/mnt/data/users/liamding/data/MTI/ppocr"
    device = "cuda"

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
    eval_mit10_all("en")  # English to all other 13 languages
    eval_mit10_all("de")  # German to all other 13 languages
    eval_mit10_all("es")  # Spanish to all other 13 languages
    eval_mit10_all("fr")  # French to all other 13 languages
    eval_mit10_all("it")  # Italian to all other 13 languages
    eval_mit10_all("ja")  # Japanese to all other 13 languages
    eval_mit10_all("pt")  # Portuguese to all other 13 languages
    eval_mit10_all("zh")  # Chinese to all other 13 languages

    # MIT10 evaluation - To evaluate specific language pairs
    #eval_mit10("en2zh")  # English to Chinese

    # MTIT6 evaluation
    eval_mtit6(lang="en2zh")
    eval_mtit6(lang="ja2zh")
    eval_mtit6(lang="ko2zh")
    eval_mtit6(lang="zh2en")
    eval_mtit6(lang="zh2ja")
    eval_mtit6(lang="zh2ko")

    # OCRMT evaluation
    eval_ocrmt()