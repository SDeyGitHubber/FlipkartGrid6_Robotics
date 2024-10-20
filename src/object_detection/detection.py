import cv2
import numpy as np
from ultralytics import YOLO
from .utils import get_color_name

def detect_and_count_objects(image_path, model, output_path='output_with_shape_color.jpg'):
    """
    Detects objects in the image, counts them, and identifies their shape and color.
    Parameters:
    - image_path: Path to the image for detection.
    - model: YOLOv8 model or any object detection model.
    - output_path: Path where the output image will be saved.
    Returns:
    - detected_info: List of dictionaries containing details of detected objects.
    - object_count: Dictionary containing counts of each detected object type.
    """
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Error: Could not load image from {image_path}")

    # Perform object detection
    results = model(img)
    detections = results[0]

    if len(detections) == 0:
        print("No detections were made.")
        return {}, {}

    object_count = {}
    detected_info = []

    for box in detections.boxes:
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
        confidence = box.conf[0]
        class_id = int(box.cls[0])
        object_name = model.names[class_id]

        if object_name not in object_count:
            object_count[object_name] = 1
        else:
            object_count[object_name] += 1

        cropped_img = img[y_min:y_max, x_min:x_max]
        shape = detect_shape(cropped_img, x_max - x_min, y_max - y_min)
        color = get_color_name(*np.mean(cropped_img, axis=(0, 1)).astype(int))

        detected_info.append({
            'bounding_box': (x_min, y_min, x_max, y_max),
            'shape': shape,
            'color': color,
            'label': object_name,
            'confidence': float(confidence)
        })

        # Draw bounding boxes and labels
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(img, f'{object_name}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the image with bounding boxes
    cv2.imwrite(output_path, img)

    # Print object count summary
    total_detected_objects = sum(object_count.values())  # Total number of objects
    count_summary = ', '.join(f"{count} {obj}(s)" for obj, count in object_count.items())
    print(f"Detected {total_detected_objects} objects - {count_summary}")

    return detected_info, object_count

def detect_shape(cropped_img, width, height):
    gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_img, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) == 3:
            return "triangle"
        elif len(approx) == 4:
            aspect_ratio = float(width) / float(height)
            return "square" if 0.9 <= aspect_ratio <= 1.1 else "rectangle"
        elif len(approx) > 4:
            return "circle"
    return "unknown"