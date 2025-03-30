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

    # Create the user message with instructions
    user_message = f"""System Instructions: {system_instructions}

Please translate the text in the image into {tgt_lang}"""

    # Skip the chat template and directly construct the prompt
    prompt = f"<|im_start|>user <image>\n{user_message} <|im_end|><|im_start|>assistant"

    # Process the image and text
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


def eval_dataset100(lang, output_name):
    """Evaluate dataset100 task with specified language direction"""
    # Get source and target language codes
    src_lang, tgt_lang_code = lang.split('2')

    # Map language code to full language name
    lang_name_map = {
        "en": "English",
        "de": "German",
        "ar": "Arabic",
        "hi": "Hindi",
        "ja": "Japanese",
        "ru": "Russian",
        "es": "Spanish"
    }
    tgt_lang = lang_name_map[tgt_lang_code]

    # Define paths based on the global variables
    image_folder = f"{root}/dataset100/test_images/"
    test_folder = Path(f"{root}/dataset100/test_100")

    # Process each JSON file in the test folder
    for json_file in test_folder.glob("*.json"):
        base_name = json_file.stem

        # Create output directory
        final_output_path = f"{output_path}/ocr_mt/{lang}/{base_name}"
        Path(final_output_path).mkdir(parents=True, exist_ok=True)

        # Load the reference data
        ref_data = json.load(open(json_file, "r", encoding="utf-8"))
        results = {}

        for img, item in tqdm(ref_data.items(), desc=f"Processing {lang} - {base_name}"):
            try:
                # Generate translation
                mt_output = generate("", [img], image_folder, tgt_lang)

                # Store results
                results[img] = {
                    "mt": mt_output,
                    "src": item["zh"]
                }
            except Exception as e:
                print(f"Error processing {img}: {e}")

        # Save results using the output_name parameter
        json.dump(results,
                  open(f"{final_output_path}/{output_name}", "w", encoding="utf-8"),
                  ensure_ascii=False,
                  indent=4)

        print(f"Completed {lang} - {base_name}. Results saved to {final_output_path}/{output_name}")


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


if __name__ == "__main__":
    # User-configurable settings
    model_id = "/mnt/data/users/liamding/data/liu_SFT/outcome_mit10m_sample500/v0-20250318-023742/checkpoint-4875-merged"  # Model path, like "output/v0-20250318-023742/checkpoint-4875"
    root = "/mnt/data/users/liamding/data/dataset"  # Base directory for all datasets, like "/mnt/data/users/liamding/data/dataset"
    output_path = "/mnt/data/users/liamding/data/liu_SFT/dataset100/evaluations/llava_onevision_random500"  # Base directory for output, like "/mnt/data/users/liamding/data/liu_SFT/dataset100/evaluations/qwen2_7b/ppocr_l_mt"
    output_name = "all.json"  # Output filename
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

    # Define language pairs to evaluate
    lang_pairs = ["zh2de", "zh2ar", "zh2hi", "zh2ja", "zh2ru", "zh2es", "zh2en"]

    # Evaluate each language pair
    for lang_pair in lang_pairs:
        print(f"Starting evaluation for {lang_pair}...")
        eval_dataset100(lang_pair, output_name)