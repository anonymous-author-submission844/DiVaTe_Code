import os
import sys
import shutil
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'AnyText'))
from PIL import Image
import torch
import numpy as np
import cv2
import re
from modelscope.pipelines import pipeline

def check_overlap_polygon(rect_pts1, rect_pts2):
    poly1 = cv2.convexHull(rect_pts1)
    poly2 = cv2.convexHull(rect_pts2)
    rect1 = cv2.boundingRect(poly1)
    rect2 = cv2.boundingRect(poly2)
    if rect1[0] + rect1[2] >= rect2[0] and rect2[0] + rect2[2] >= rect1[0] and rect1[1] + rect1[3] >= rect2[1] and rect2[1] + rect2[3] >= rect1[1]:
        return True
    return False

def draw_rects(width, height, rects):
    img = np.zeros((height, width, 1), dtype=np.uint8)
    for rect in rects:
        x1 = int(rect[0] * width)
        y1 = int(rect[1] * height)
        w = int(rect[2] * width)
        h = int(rect[3] * height)
        x2 = x1 + w
        y2 = y1 + h
        cv2.rectangle(img, (x1, y1), (x2, y2), 255, -1)
    return img

def count_lines(prompt):
    prompt = prompt.replace('“', '"')
    prompt = prompt.replace('”', '"')
    p = '"(.*?)"'
    strs = re.findall(p, prompt)
    if len(strs) == 0:
        strs = [' ']
    return len(strs)

def generate_rectangles(w, h, n, max_trys=200):
    img = np.zeros((h, w, 1), dtype=np.uint8)
    rectangles = []
    attempts = 0
    n_pass = 0
    low_edge = int(max(w, h)*0.3 if n <= 3 else max(w, h)*0.2)  # ~150, ~100
    while attempts < max_trys:
        rect_w = min(np.random.randint(max((w*0.5)//n, low_edge), w), int(w*0.8))
        ratio = np.random.uniform(4, 10)
        rect_h = max(low_edge, int(rect_w/ratio))
        rect_h = min(rect_h, int(h*0.8))
        # gen rotate angle
        rotation_angle = 0
        rand_value = np.random.rand()
        if rand_value < 0.7:
            pass
        elif rand_value < 0.8:
            rotation_angle = np.random.randint(0, 40)
        elif rand_value < 0.9:
            rotation_angle = np.random.randint(140, 180)
        else:
            rotation_angle = np.random.randint(85, 95)
        # rand position
        x = np.random.randint(0, w - rect_w)
        y = np.random.randint(0, h - rect_h)
        # get vertex
        rect_pts = cv2.boxPoints(((rect_w/2, rect_h/2), (rect_w, rect_h), rotation_angle))
        rect_pts = np.int32(rect_pts)
        # move
        rect_pts += (x, y)
        # check boarder
        if np.any(rect_pts < 0) or np.any(rect_pts[:, 0] >= w) or np.any(rect_pts[:, 1] >= h):
            attempts += 1
            continue
        # check overlap
        if any(check_overlap_polygon(rect_pts, rp) for rp in rectangles):
            attempts += 1
            continue
        n_pass += 1
        cv2.fillPoly(img, [rect_pts], 255)
        rectangles.append(rect_pts)
        if n_pass == n:
            break
    print("attempts:", attempts)
    if len(rectangles) != n:
        raise Exception(f'Failed in auto generate positions after {attempts} attempts, try again!')
    return img

class AnyText:
    def __init__(self, model_path=None):
        # Get the absolute path to the AnyText2 directory
        anytext_dir = os.path.join(os.path.dirname(__file__), '..', 'AnyText')
        current_dir = os.getcwd()
        
        # Create necessary directories if they don't exist
        os.makedirs(os.path.join(current_dir, 'ocr_recog'), exist_ok=True)
        os.makedirs(os.path.join(current_dir, 'models_yaml'), exist_ok=True)
        
        # Copy necessary files
        shutil.copy2(
            os.path.join(anytext_dir, 'ocr_recog', 'ppocr_keys_v1.txt'),
            os.path.join(current_dir, 'ocr_recog', 'ppocr_keys_v1.txt')
        )
        shutil.copy2(
            os.path.join(anytext_dir, 'models_yaml', 'anytext_sd15.yaml'),
            os.path.join(current_dir, 'models_yaml', 'anytext_sd15.yaml')
        )
        self.pipe = pipeline('my-anytext-task', model=model_path if model_path else 'damo/cv_anytext_text_generation_editing',
                            use_translator=False, font_path='font/Arial_Unicode.ttf',
                            model_revision='v1.1.3')

    def get_sort_priority(self, prompt: str):
        if "up" in prompt or "top" in prompt or "front" in prompt or "north" in prompt:
            return '↕'
        elif "next" in prompt or "left" in prompt or "east" in prompt:
            return '↔'
        return None

    def generate_image(self, prompt: str, generator: torch.Generator, steps=20, guidance_scale=7.5) -> Image.Image:
        prompt = prompt.replace("'",'"')
        n_lines = count_lines(prompt)
        pos_imgs = generate_rectangles(512, 512, n_lines, max_trys=10000)
        mode = 'text-generation'

        params = {
            "show_debug": True,
            "image_count": 1,
            "ddim_steps": 20,
        }

        if n_lines > 1:
            params['sort_priority'] = self.get_sort_priority(prompt)
        
        input_data = {
            "prompt": prompt,
            "seed": generator.seed(),
            "draw_pos": pos_imgs
        }
        results, rtn_code, rtn_warning, debug_info = self.pipe(input_data, mode=mode, **params)
        return Image.fromarray(results[0])