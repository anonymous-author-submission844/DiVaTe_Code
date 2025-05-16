import os
import json
import torch
import argparse
import shutil
import ast
from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def preprocess_image(image_path):
    """Preprocess image for better OCR results"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    img = clahe.apply(img)

    # Sharpen image
    kernel_sharp = np.array([[0, -1, 0],
                            [-1, 5,-1],
                            [0, -1, 0]])
    img_sharp = cv2.filter2D(img, -1, kernel_sharp)

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(img_sharp, 255, 
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 
                                 15, 7)

    return binary


def perform_ocr_paddle(image_path, ocr_model):
    """Perform OCR on an image file using PaddleOCR.
    
    Args:
        image_path: Path to the saved image file
        ocr_model: PaddleOCR model instance
        
    Returns:
        List of OCR results
    """
    # Preprocess image
    preprocessed_img = preprocess_image(image_path)
    
    # Perform OCR
    result = ocr_model.ocr(preprocessed_img, cls=True)
    
    # Format OCR results
    ocr_data = []
    if result[0]:  # Check if results exist
        for line in result[0]:
            bbox, (text, conf) = line
            # Convert numpy values to Python native types
            bbox = [[int(x), int(y)] for x, y in bbox]
            ocr_data.append({
                "text": text,
                "confidence": float(conf),
                "bbox": bbox
            })
    
    return ocr_data


def perform_ocr_easy(image_path, ocr_model):
    """Perform OCR on an image file using EasyOCR.
    
    Args:
        image_path: Path to the saved image file
        ocr_model: EasyOCR model instance
        
    Returns:
        List of OCR results
    """
    # Preprocess image
    # preprocessed_img = preprocess_image(image_path)
    
    # Perform OCR
    results = ocr_model.readtext(image_path)
    
    # Format OCR results
    ocr_data = []
    for result in results:
        bbox, text, conf = result
        # Convert numpy values to Python native types
        bbox = [[int(x), int(y)] for x, y in bbox]
        ocr_data.append({
            "text": text,
            "confidence": float(conf),
            "bbox": bbox
        })
    
    return ocr_data


def perform_ocr_qwen(image_path, ocr_model):
    """Perform OCR on an image file using Qwen-VL.
    
    Args:
        image_path: Path to the saved image file
        ocr_model: Tuple of (model, processor) for Qwen-VL
        
    Returns:
        List of OCR results
    """
    model, processor = ocr_model
    prompt = "Spotting all the text in the image with line-level, and output in JSON format."
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(torch.float16)
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Parse the JSON output
    try:
        # Extract JSON string from the output
        json_str = output_text.split('```json')[1].split('```')[0].strip()
        results = ast.literal_eval(json_str)
        
        # Convert to our standard format
        ocr_data = []
        for result in results:
            bbox = result["bbox_2d"]
            # Convert [x1, y1, x2, y2] to [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            bbox_points = [
                [bbox[0], bbox[1]],  # top-left
                [bbox[2], bbox[1]],  # top-right
                [bbox[2], bbox[3]],  # bottom-right
                [bbox[0], bbox[3]]   # bottom-left
            ]
            ocr_data.append({
                "text": result["text_content"],
                "confidence": 1.0,  # Qwen doesn't provide confidence scores
                "bbox": bbox_points
            })
        return ocr_data
    except Exception as e:
        print(f"Error parsing Qwen-VL output: {e}")
        print(f"Output text: {output_text}")
        return []


def mask_text_regions(image_path, ocr_results, ocr_type):
    """Mask text regions with black rectangles based on OCR bounding boxes.
    
    Args:
        image_path: Path to the original image
        ocr_results: List of OCR results containing bounding boxes
        ocr_type: Type of OCR used ('paddle', 'easy', or 'qwen')
        
    Returns:
        PIL Image with text regions masked
    """
    # Open the image
    image = Image.open(image_path)
    width, height = image.size
    
    # Create a drawing context
    draw = ImageDraw.Draw(image)
    
    # Mask each detected text region with black
    for result in ocr_results:
        bbox = result["bbox"]
        if ocr_type == 'paddle':
            # PaddleOCR bbox format
            points = [(int(bbox[i][0]), int(bbox[i][1])) for i in range(len(bbox))]
        else:
            # EasyOCR and Qwen-VL bbox format
            points = [(int(x), int(y)) for x, y in bbox]
        
        # Get min and max coordinates
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Calculate expansion margin (10% of the width and height)
        margin_x = int((max_x - min_x) * 0.1)
        margin_y = int((max_y - min_y) * 0.1)
        
        # Expand the bounding box
        min_x = max(0, min_x - margin_x)
        max_x = min(width, max_x + margin_x)
        min_y = max(0, min_y - margin_y)
        max_y = min(height, max_y + margin_y)
        
        # Create expanded rectangle points
        expanded_points = [
            (min_x, min_y),  # top-left
            (max_x, min_y),  # top-right
            (max_x, max_y),  # bottom-right
            (min_x, max_y)   # bottom-left
        ]
        
        # Draw a filled black polygon over the expanded text area
        draw.polygon(expanded_points, fill="black")
    
    return image


def main():
    parser = argparse.ArgumentParser(description="Run OCR on the generated images")
    parser.add_argument("--output_dir", type=str, required=True, help="Main output directory containing all subdirectories")
    parser.add_argument("--ocr_type", type=str, choices=['paddle', 'easy', 'qwen', 'qwen_awq'], default='paddle', 
                       help="Type of OCR to use: 'paddle' for PaddleOCR, 'easy' for EasyOCR, or 'qwen' for Qwen-VL, 'qwen_awq' for Qwen-VL-AWQ")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model")
    # parser.add_argument("--quantized", action='store_true', help="Whether to use quantized models")
    args = parser.parse_args()
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define subdirectories
    generated_images_path = os.path.join(args.output_dir, "generated_images")
    ocr_results_path = os.path.join(args.output_dir, "ocr_results")
    masked_images_path = os.path.join(args.output_dir, "masked_images")
    
    # Make sure generated_images directory exists
    if not os.path.exists(generated_images_path):
        print(f"Error: Generated images directory {generated_images_path} does not exist.")
        return
        
    # Create other subdirectories
    os.makedirs(ocr_results_path, exist_ok=True)
    os.makedirs(masked_images_path, exist_ok=True)

    # Initialize OCR model
    print(f"Initializing {args.ocr_type} OCR model...")
    if args.ocr_type == 'paddle':
        from paddleocr import PaddleOCR
        ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
        perform_ocr = perform_ocr_paddle
    elif args.ocr_type == 'easy':
        import easyocr
        ocr_model = easyocr.Reader(['en'])
        perform_ocr = perform_ocr_easy
    elif args.ocr_type == 'qwen':        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path if args.model_path else "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(args.model_path if args.model_path else "Qwen/Qwen2.5-VL-7B-Instruct")
        ocr_model = (model, processor)
        perform_ocr = perform_ocr_qwen
    elif args.ocr_type == 'qwen_awq':
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path if args.model_path else "Qwen/Qwen2.5-VL-7B-Instruct-AWQ", torch_dtype=torch.float16, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(args.model_path if args.model_path else "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
        ocr_model = (model, processor)
        perform_ocr = perform_ocr_qwen

    # Dictionary to collect all OCR results
    all_ocr_results = {}

    # Process each image in the input directory
    for image_file in tqdm(os.listdir(generated_images_path)):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(generated_images_path, image_file)
            ocr_results = perform_ocr(image_path, ocr_model)
            
            # Store results with the image filename as key
            all_ocr_results[image_file] = {
                "path": image_path,
                "ocr_results": ocr_results
            }
            
            # Create masked image path
            masked_image_path = os.path.join(masked_images_path, f"masked_{image_file}")
            
            # Mask text regions and save if OCR results exist, otherwise copy original image
            if ocr_results:
                masked_image = mask_text_regions(image_path, ocr_results, args.ocr_type)
                masked_image.save(masked_image_path)
            else:
                # If no OCR results, copy the original image
                shutil.copy2(image_path, masked_image_path)
                
            all_ocr_results[image_file]["masked_path"] = masked_image_path
            print(f"Processed {image_file} with {len(ocr_results)} OCR results")
    
    # Save all OCR results at once
    results_file = os.path.join(ocr_results_path, "ocr_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_ocr_results, f, indent=4)
    
    print(f"All OCR results saved to {results_file}")
    print(f"Masked images saved to {masked_images_path}")

if __name__ == "__main__":
    main()