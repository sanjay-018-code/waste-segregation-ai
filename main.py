import cv2
import time
import json
import logging
import numpy as np
from ultralytics import YOLO
import sys
import requests
import zipfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger('WasteIntelligence')
logger.setLevel(logging.WARNING)

print("Initializing System...")
print("Loading Fast Model (yolov8n.pt)...")
try:
    model_fast = YOLO('yolov8n.pt') 
    model_accurate = None # Deferred loading
    print("Loading Material Classifier (yolov8m-cls.pt)...")
    classifier = YOLO('yolov8m-cls.pt')
    if os.path.exists('runs/classify/train/weights/best.pt'):
        classifier = YOLO('runs/classify/train/weights/best.pt')
        print("Loaded custom trained material classifier.")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)
    
current_model = model_fast
model_mode = "fast"
tracking_enabled = False

# Comprehensive Mapping for ALL 80 COCO Classes
WASTE_MAPPING = {
    # Living / Animals
    'person': {'material': 'Biological', 'category': 'Non-Waste', 'bin': 'Do Not Dispose', 'color': (200, 200, 200)},
    'bird': {'material': 'Biological', 'category': 'Non-Waste', 'bin': 'Do Not Dispose', 'color': (200, 200, 200)},
    'cat': {'material': 'Biological', 'category': 'Non-Waste', 'bin': 'Do Not Dispose', 'color': (200, 200, 200)},
    'dog': {'material': 'Biological', 'category': 'Non-Waste', 'bin': 'Do Not Dispose', 'color': (200, 200, 200)},
    'horse': {'material': 'Biological', 'category': 'Non-Waste', 'bin': 'Do Not Dispose', 'color': (200, 200, 200)},
    'sheep': {'material': 'Biological', 'category': 'Non-Waste', 'bin': 'Do Not Dispose', 'color': (200, 200, 200)},
    'cow': {'material': 'Biological', 'category': 'Non-Waste', 'bin': 'Do Not Dispose', 'color': (200, 200, 200)},
    'elephant': {'material': 'Biological', 'category': 'Non-Waste', 'bin': 'Do Not Dispose', 'color': (200, 200, 200)},
    'bear': {'material': 'Biological', 'category': 'Non-Waste', 'bin': 'Do Not Dispose', 'color': (200, 200, 200)},
    'zebra': {'material': 'Biological', 'category': 'Non-Waste', 'bin': 'Do Not Dispose', 'color': (200, 200, 200)},
    'giraffe': {'material': 'Biological', 'category': 'Non-Waste', 'bin': 'Do Not Dispose', 'color': (200, 200, 200)},

    # Vehicles / Infrastructure
    'bicycle': {'material': 'Metal/Rubber', 'category': 'Recyclable/Bulky', 'bin': 'Scrap Metal', 'color': (150, 150, 150)},
    'car': {'material': 'Metal/Mixed', 'category': 'Bulky', 'bin': 'Scrap Yard', 'color': (150, 150, 150)},
    'motorcycle': {'material': 'Metal/Mixed', 'category': 'Bulky', 'bin': 'Scrap Yard', 'color': (150, 150, 150)},
    'airplane': {'material': 'Metal/Mixed', 'category': 'Bulky', 'bin': 'Scrap Yard', 'color': (150, 150, 150)},
    'bus': {'material': 'Metal/Mixed', 'category': 'Bulky', 'bin': 'Scrap Yard', 'color': (150, 150, 150)},
    'train': {'material': 'Metal/Mixed', 'category': 'Bulky', 'bin': 'Scrap Yard', 'color': (150, 150, 150)},
    'truck': {'material': 'Metal/Mixed', 'category': 'Bulky', 'bin': 'Scrap Yard', 'color': (150, 150, 150)},
    'boat': {'material': 'Metal/Wood/Plastic', 'category': 'Bulky', 'bin': 'Scrap Yard', 'color': (150, 150, 150)},
    'traffic light': {'material': 'Metal/Electronics', 'category': 'Hazardous', 'bin': 'E-Waste Dropoff', 'color': (0, 0, 255)},
    'fire hydrant': {'material': 'Metal', 'category': 'Recyclable/Bulky', 'bin': 'Scrap Metal', 'color': (150, 150, 150)},
    'stop sign': {'material': 'Metal', 'category': 'Recyclable', 'bin': 'Scrap Metal', 'color': (150, 150, 150)},
    'parking meter': {'material': 'Metal/Electronics', 'category': 'Hazardous', 'bin': 'E-Waste Dropoff', 'color': (0, 0, 255)},

    # Textiles / Accessories
    'backpack': {'material': 'Textile', 'category': 'Reusable', 'bin': 'Textile Donation', 'color': (100, 100, 150)},
    'umbrella': {'material': 'Mixed/Textile', 'category': 'Landfill', 'bin': 'Black Bin', 'color': (150, 150, 150)},
    'handbag': {'material': 'Leather/Textile', 'category': 'Reusable', 'bin': 'Textile Donation', 'color': (100, 100, 150)},
    'tie': {'material': 'Textile', 'category': 'Reusable', 'bin': 'Textile Donation', 'color': (100, 100, 150)},
    'suitcase': {'material': 'Mixed/Plastic', 'category': 'Bulky', 'bin': 'Bulky Waste', 'color': (150, 150, 150)},

    # Sports / Leisure
    'frisbee': {'material': 'Plastic', 'category': 'Recyclable', 'bin': 'Blue Bin', 'color': (255, 150, 50)},
    'skis': {'material': 'Mixed', 'category': 'Bulky', 'bin': 'Bulky Waste', 'color': (150, 150, 150)},
    'snowboard': {'material': 'Mixed', 'category': 'Bulky', 'bin': 'Bulky Waste', 'color': (150, 150, 150)},
    'sports ball': {'material': 'Rubber/Leather', 'category': 'Landfill', 'bin': 'Black Bin', 'color': (150, 150, 150)},
    'kite': {'material': 'Paper/Plastic', 'category': 'Landfill', 'bin': 'Black Bin', 'color': (150, 150, 150)},
    'baseball bat': {'material': 'Wood/Metal', 'category': 'Recyclable', 'bin': 'Blue Bin', 'color': (255, 150, 50)},
    'baseball glove': {'material': 'Leather', 'category': 'Landfill', 'bin': 'Black Bin', 'color': (150, 150, 150)},
    'skateboard': {'material': 'Wood/Plastic', 'category': 'Landfill', 'bin': 'Black Bin', 'color': (150, 150, 150)},
    'surfboard': {'material': 'Foam/Fiberglass', 'category': 'Bulky', 'bin': 'Bulky Waste', 'color': (150, 150, 150)},
    'tennis racket': {'material': 'Metal/Plastic', 'category': 'Landfill', 'bin': 'Black Bin', 'color': (150, 150, 150)},

    # Kitchenware / Food Containers
    'bottle': {'material': 'Plastic/Glass', 'category': 'Recyclable', 'bin': 'Blue Bin', 'color': (255, 150, 50)},
    'wine glass': {'material': 'Glass', 'category': 'Recyclable', 'bin': 'Blue Bin', 'color': (255, 150, 50)},
    'cup': {'material': 'Paper/Plastic', 'category': 'Recyclable', 'bin': 'Blue Bin', 'color': (255, 150, 50)},
    'fork': {'material': 'Metal/Plastic', 'category': 'Recyclable', 'bin': 'Blue Bin', 'color': (255, 150, 50)},
    'knife': {'material': 'Metal', 'category': 'Recyclable', 'bin': 'Blue Bin', 'color': (255, 150, 50)},
    'spoon': {'material': 'Metal/Plastic', 'category': 'Recyclable', 'bin': 'Blue Bin', 'color': (255, 150, 50)},
    'scale': {'material': 'Metal/Plastic', 'category': 'Recyclable', 'bin': 'Blue Bin', 'color': (255, 150, 50)},
    'bowl': {'material': 'Ceramic/Plastic', 'category': 'Landfill', 'bin': 'Black Bin', 'color': (150, 150, 150)},
    'vase': {'material': 'Glass/Ceramic', 'category': 'Recyclable', 'bin': 'Blue Bin', 'color': (255, 150, 50)},

    # Organic / Food
    'banana': {'material': 'Organic', 'category': 'Compostable', 'bin': 'Green Bin', 'color': (0, 255, 0)},
    'apple': {'material': 'Organic', 'category': 'Compostable', 'bin': 'Green Bin', 'color': (0, 255, 0)},
    'sandwich': {'material': 'Organic', 'category': 'Compostable', 'bin': 'Green Bin', 'color': (0, 255, 0)},
    'orange': {'material': 'Organic', 'category': 'Compostable', 'bin': 'Green Bin', 'color': (0, 255, 0)},
    'broccoli': {'material': 'Organic', 'category': 'Compostable', 'bin': 'Green Bin', 'color': (0, 255, 0)},
    'carrot': {'material': 'Organic', 'category': 'Compostable', 'bin': 'Green Bin', 'color': (0, 255, 0)},
    'hot dog': {'material': 'Organic', 'category': 'Compostable', 'bin': 'Green Bin', 'color': (0, 255, 0)},
    'pizza': {'material': 'Organic', 'category': 'Compostable', 'bin': 'Green Bin', 'color': (0, 255, 0)},
    'donut': {'material': 'Organic', 'category': 'Compostable', 'bin': 'Green Bin', 'color': (0, 255, 0)},
    'cake': {'material': 'Organic', 'category': 'Compostable', 'bin': 'Green Bin', 'color': (0, 255, 0)},

    # Furniture / Bulky
    'chair': {'material': 'Wood/Metal', 'category': 'Bulky', 'bin': 'Bulky Waste', 'color': (150, 100, 50)},
    'couch': {'material': 'Textile/Wood', 'category': 'Bulky', 'bin': 'Bulky Waste', 'color': (150, 100, 50)},
    'potted plant': {'material': 'Organic/Ceramic', 'category': 'Compostable', 'bin': 'Green Bin', 'color': (0, 255, 0)},
    'bed': {'material': 'Textile/Wood', 'category': 'Bulky', 'bin': 'Bulky Waste', 'color': (150, 100, 50)},
    'dining table': {'material': 'Wood/Metal', 'category': 'Bulky', 'bin': 'Bulky Waste', 'color': (150, 100, 50)},
    'toilet': {'material': 'Ceramic', 'category': 'Bulky', 'bin': 'Bulky Waste', 'color': (150, 100, 50)},
    'bench': {'material': 'Wood/Metal', 'category': 'Bulky', 'bin': 'Bulky Waste', 'color': (150, 100, 50)},

    # E-Waste / Electronics``
    'tv': {'material': 'E-Waste', 'category': 'Hazardous', 'bin': 'E-Waste Dropoff', 'color': (0, 0, 255)},
    'laptop': {'material': 'E-Waste', 'category': 'Hazardous', 'bin': 'E-Waste Dropoff', 'color': (0, 0, 255)},
    'mouse': {'material': 'E-Waste', 'category': 'Hazardous', 'bin': 'E-Waste Dropoff', 'color': (0, 0, 255)},
    'remote': {'material': 'E-Waste', 'category': 'Hazardous', 'bin': 'E-Waste Dropoff', 'color': (0, 0, 255)},
    'keyboard': {'material': 'E-Waste', 'category': 'Hazardous', 'bin': 'E-Waste Dropoff', 'color': (0, 0, 255)},
    'cell phone': {'material': 'E-Waste', 'category': 'Hazardous', 'bin': 'E-Waste Dropoff', 'color': (0, 0, 255)},
    'microwave': {'material': 'E-Waste/Metal', 'category': 'Hazardous', 'bin': 'E-Waste Dropoff', 'color': (0, 0, 255)},
    'oven': {'material': 'E-Waste/Metal', 'category': 'Hazardous', 'bin': 'E-Waste Dropoff', 'color': (0, 0, 255)},
    'toaster': {'material': 'E-Waste', 'category': 'Hazardous', 'bin': 'E-Waste Dropoff', 'color': (0, 0, 255)},
    'refrigerator': {'material': 'E-Waste/Metal', 'category': 'Hazardous/Bulky', 'bin': 'E-Waste Dropoff', 'color': (0, 0, 255)},
    'hair drier': {'material': 'E-Waste', 'category': 'Hazardous', 'bin': 'E-Waste Dropoff', 'color': (0, 0, 255)},

    # Misc Objects
    'sink': {'material': 'Metal/Ceramic', 'category': 'Bulky', 'bin': 'Bulky Waste', 'color': (150, 100, 50)},
    'book': {'material': 'Paper', 'category': 'Recyclable', 'bin': 'Blue Bin', 'color': (255, 150, 50)},
    'clock': {'material': 'Mixed/E-Waste', 'category': 'Landfill', 'bin': 'Black Bin', 'color': (150, 150, 150)},
    'scissors': {'material': 'Metal', 'category': 'Recyclable', 'bin': 'Blue Bin', 'color': (255, 150, 50)},
    'teddy bear': {'material': 'Textile', 'category': 'Landfill', 'bin': 'Black Bin', 'color': (150, 150, 150)},
    'toothbrush': {'material': 'Plastic', 'category': 'Landfill', 'bin': 'Black Bin', 'color': (150, 150, 150)}
}

MATERIAL_KEYWORDS = {
    'Plastic': ['plastic', 'bottle', 'container', 'cup', 'bag', 'toy', 'utensil', 'polyethylene', 'pvc'],
    'Metal': ['metal', 'can', 'aluminum', 'steel', 'iron', 'tool', 'utensil', 'copper', 'brass'],
    'Glass': ['glass', 'jar', 'bottle', 'window', 'mirror'],
    'Paper': ['paper', 'cardboard', 'newspaper', 'book', 'magazine', 'card'],
    'Organic': ['fruit', 'vegetable', 'food', 'banana', 'apple', 'bread', 'meat', 'dairy'],
    'Wood': ['wood', 'lumber', 'timber', 'furniture', 'plywood'],
    'Textile': ['fabric', 'cloth', 'textile', 'cotton', 'shirt', 'pants', 'wool', 'silk'],
    'Ceramic': ['ceramic', 'pottery', 'plate', 'bowl', 'tile'],
    'E-Waste': ['electronic', 'computer', 'phone', 'device', 'battery', 'circuit'],
    'Rubber': ['rubber', 'tire', 'ball', 'elastic'],
    'Foam': ['foam', 'styrofoam', 'sponge'],
    'Leather': ['leather', 'hide', 'skin'],
    'Concrete': ['concrete', 'cement', 'stone'],
    'Composite': ['composite', 'mixed', 'alloy']
}

DATASET_MATERIAL_MAP = {
    'glass': 'Glass',
    'paper': 'Paper',
    'cardboard': 'Paper',
    'plastic': 'Plastic',
    'metal': 'Metal',
    'trash': 'Unknown'
}

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

def classify_material_from_class(predicted_class):
    predicted_lower = predicted_class.lower()
    for material, keywords in MATERIAL_KEYWORDS.items():
        if any(keyword in predicted_lower for keyword in keywords):
            return material
    return 'Unknown'

def dataset_has_images(dataset_root):
    if not os.path.isdir(dataset_root):
        return False
    for _, _, filenames in os.walk(dataset_root):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
                return True
    return False

def download_dataset(dataset_path):
    dataset_root = find_dataset_root(dataset_path)
    if dataset_has_images(dataset_root):
        return True
    print("Downloading TrashNet dataset for material classification...")
    url = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
    try:
        response = requests.get(url, timeout=300)
        with open('dataset.zip', 'wb') as f:
            f.write(response.content)
        with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        print("Dataset downloaded and extracted.")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


def train_model():
    dataset_path = 'dataset'
    if not download_dataset(dataset_path):
        print("Could not download dataset. Training aborted.")
        return
    dataset_root = find_dataset_root(dataset_path)
    print("Training custom material classifier...")
    model = YOLO('yolov8m-cls.pt')
    results = model.train(data=dataset_root, epochs=20, imgsz=224, device=0)
    print("Training complete. Custom model saved in runs/classify/train/weights/best.pt")
    # Update the classifier to use the trained model
    global classifier
    classifier = YOLO('runs/classify/train/weights/best.pt')


def find_dataset_root(dataset_path):
    # TrashNet zip commonly contains a top-level "dataset-resized/" folder. In the original
    # repository layout it may also appear under "data/dataset-resized/".
    candidates = [
        os.path.join(dataset_path, 'data', 'dataset-resized'),
        os.path.join(dataset_path, 'dataset-resized'),
        os.path.join(dataset_path, 'data'),
        dataset_path,
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return dataset_path


def collect_dataset_images(dataset_path):
    root = find_dataset_root(dataset_path)
    images = []
    for class_name in sorted(os.listdir(root)):
        class_folder = os.path.join(root, class_name)
        if not os.path.isdir(class_folder):
            continue
        for filename in sorted(os.listdir(class_folder)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                images.append((os.path.join(class_folder, filename), class_name))
    return images


def predict_material_from_image(img):
    results = classifier.predict(img, verbose=False)
    probs = np.array(results[0].probs)
    top_indices = np.argsort(-probs)[:5]
    for idx in top_indices:
        candidate = results[0].names[int(idx)]
        material = classify_material_from_class(candidate)
        if material != 'Unknown':
            return material
    return 'Unknown'


def validate_model():
    dataset_path = 'dataset'
    if not download_dataset(dataset_path):
        print("Validation failed because the dataset could not be downloaded.")
        return

    root = find_dataset_root(dataset_path)
    images = collect_dataset_images(dataset_path)
    if not images:
        print(f"No validation images found in {root}")
        return

    custom_model = os.path.exists('runs/classify/train/weights/best.pt')
    if custom_model:
        print("Validating custom trained classifier...")
    else:
        print("Validating base classifier with material keyword mapping...")

    total = 0
    correct = 0
    material_correct = 0
    class_counts = {}
    material_counts = {}

    for image_path, true_class in images:
        img = cv2.imread(image_path)
        if img is None:
            continue
        total += 1
        results = classifier.predict(img, verbose=False)
        probs = np.array(results[0].probs)
        top_idx = int(np.argmax(probs))
        predicted_name = results[0].names[top_idx]

        if custom_model:
            predicted_class = predicted_name
            if predicted_class == true_class:
                correct += 1
            predicted_material = DATASET_MATERIAL_MAP.get(predicted_class, 'Unknown')
            true_material = DATASET_MATERIAL_MAP.get(true_class, 'Unknown')
            if predicted_material == true_material:
                material_correct += 1
        else:
            predicted_material = predict_material_from_image(img)
            true_material = DATASET_MATERIAL_MAP.get(true_class, 'Unknown')
            if predicted_material == true_material:
                material_correct += 1

        class_counts[true_class] = class_counts.get(true_class, 0) + 1
        material_counts[true_material] = material_counts.get(true_material, 0) + 1

    print(f"Validation images: {total}")
    if custom_model:
        print(f"Top-1 class accuracy: {correct/total:.2%} ({correct}/{total})")
    print(f"Material accuracy: {material_correct/total:.2%} ({material_correct}/{total})")
    print("Per-class counts:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count}")

    if custom_model:
        print("Custom model validation complete.")
    else:
        print("Base classifier validation complete.")


def analyze_frame(results, frame):
    detected_items = []
    category_counts = {}
    
    # We will grab every single object detected so you can see it working!
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = results[0].names[cls_id]
        
        # Crop bounding box for material classification
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        
        material = 'Unknown'
        if crop.size > 0:
            try:
                class_results = classifier.predict(crop, verbose=False)
                predicted_classes = [class_results[0].names[i] for i in class_results[0].probs.top5]
                material = 'Unknown'
                for pred in predicted_classes:
                    mat = classify_material_from_class(pred)
                    if mat != 'Unknown':
                        material = mat
                        break
            except:
                material = 'Unknown'
        
        if class_name in WASTE_MAPPING:
            info = WASTE_MAPPING[class_name]
            # Use classified material if available, else keep mapped
            if material != 'Unknown':
                info = info.copy()
                info['material'] = material
            item = {
                'name': class_name.title(),
                'material': info['material'],
                'recyclability': info['category'],
                'confidence': round(conf * 100, 1),
                'bin': info['bin'],
                'color': info['color'],
                'biodegradable': 'Yes' if info['category'] == 'Compostable' else 'No',
                'bbox': (x1, y1, x2, y2)
            }
        else:
            # For objects like 'person', 'metal plate' (if misclassified), etc.
            item = {
                'name': class_name.title() + " (Generic)",
                'material': material,
                'recyclability': 'Unknown',
                'confidence': round(conf * 100, 1),
                'bin': 'Awaiting Manual Classification',
                'color': (255, 255, 255),
                'biodegradable': 'Unknown',
                'bbox': (x1, y1, x2, y2)
            }
            
        detected_items.append(item)
        
        cat = item['recyclability']
        if cat != 'Unknown':
            category_counts[cat] = category_counts.get(cat, 0) + 1

    dominant = "Mixed/None"
    if category_counts:
        dominant = max(category_counts, key=category_counts.get)
        
    return detected_items, dominant

def draw_overlay(frame, detected_items, dominant):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    panel_width = 380
    cv2.rectangle(overlay, (w - panel_width, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Header
    cv2.putText(frame, "SMART MATERIAL CLASSIFIER", (w - panel_width + 15, 35), font, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Time: {time.strftime('%H:%M:%S')}", (w - panel_width + 15, 65), font, 0.5, (200, 200, 200), 1)
    
    # Stats
    cv2.putText(frame, f"Total Materials: {len(detected_items)}", (w - panel_width + 15, 110), font, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Dominant Material: {dominant}", (w - panel_width + 15, 140), font, 0.6, (255, 255, 255), 1)
    
    # Guidance
    y_offset = 190
    cv2.putText(frame, "Material Disposal:", (w - panel_width + 15, y_offset), font, 0.7, (0, 255, 0), 2)
    y_offset += 30
    
    guidance_given = set()
    for item in detected_items:
        if item['bin'] != 'Awaiting Manual Classification' and item['material'] not in guidance_given:
            cv2.putText(frame, f"-> {item['material']}: {item['bin']}", (w - panel_width + 15, y_offset), font, 0.5, item['color'], 1)
            y_offset += 25
            guidance_given.add(item['material'])
            
    if not guidance_given:
        cv2.putText(frame, "-> No specific action required", (w - panel_width + 15, y_offset), font, 0.5, (150, 150, 150), 1)
        
    y_offset += 40
    cv2.putText(frame, "Detected Materials:", (w - panel_width + 15, y_offset), font, 0.6, (0, 255, 255), 1)
    y_offset += 25
    
    for i, item in enumerate(detected_items[:8]): 
        text_line = f"{item['material']} (Bio: {item['biodegradable']})"
        cv2.putText(frame, text_line, (w - panel_width + 15, y_offset), font, 0.5, item['color'], 1)
        y_offset += 25

def main():
    global current_model, model_mode, tracking_enabled
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            train_model()
            exit()
        elif sys.argv[1] == 'validate':
            validate_model()
            exit()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Make sure another app isn't using it!")
        return

    # Try to set high resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n--- Smart Waste Detection Active ---")
    print("Controls:")
    print("  M - Load accurate model (yolov8m.pt) [Takes 30s]")
    print("  T - Toggle Tracking")
    print("  C - Capture screenshot")
    print("  Q - Quit")
    print("  (Use command line modes instead of in-app: 'python main.py train' or 'python main.py validate')")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # We lowered confidence to 0.25 to make sure it picks up generic objects for testing!
        # And we DO NOT filter classes, so it shows exactly what it thinks it sees.
        if tracking_enabled:
            results = current_model.track(frame, persist=True, verbose=False, conf=0.25)
        else:
            results = current_model.predict(frame, verbose=False, conf=0.25)
            
        # YOLO plots boxes for everything it found
        annotated_frame = frame.copy()
        
        # Analyze our specific mappings
        detected_items, dominant = analyze_frame(results, frame)
        
        # Draw material boxes and labels
        for det in detected_items:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), det['color'], 2)
            label = f"{det['material']} (Bio: {det['biodegradable']})"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, det['color'], 2)
        
        # Draw HUD dashboard
        draw_overlay(annotated_frame, detected_items, dominant)
        
        # Overlay current state
        cv2.putText(annotated_frame, f"Model: {model_mode.upper()} | Track: {'ON' if tracking_enabled else 'OFF'}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
        cv2.imshow("Smart Waste Vision", annotated_frame)
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Shutting down...")
            break
        elif key == ord('c'):
            filename = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"Saved: {filename}")
        elif key == ord('t'):
            tracking_enabled = not tracking_enabled
            print(f"Tracking toggled: {tracking_enabled}")
        elif key == ord('m'):
            global model_accurate
            if model_mode == "fast":
                if model_accurate is None:
                    print("DOWNLOADING/LOADING yolov8m.pt... Please wait, camera is paused!")
                    model_accurate = YOLO('yolov8m.pt')
                model_mode = "accurate"
                current_model = model_accurate
            else:
                model_mode = "fast"
                current_model = model_fast
            print(f"Switched to model: {model_mode}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
