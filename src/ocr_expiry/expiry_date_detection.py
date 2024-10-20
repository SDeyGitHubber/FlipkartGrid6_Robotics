import requests
from PIL import Image
import pytesseract
import re
import os
from inference_sdk import InferenceHTTPClient

# Load environment variables if using .env (for API key hiding)
from dotenv import load_dotenv
load_dotenv()

# Initialize the Inference Client (Replace API_KEY with environment variable for security)
API_KEY = os.getenv("INFERENCE_API_KEY")

if not API_KEY:
    raise ValueError("API key not found. Please set the ROBOFLOW_API_KEY in your environment.")

CLIENT = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=API_KEY)

# Function to perform inference and extract expiry date
def detect_expiry_date(image_path, model_id="flipkartgrid-ocr-tmd6q-lbde9/1"):
    """
    Detect expiry dates using Roboflow inference API and perform OCR to extract date text.

    Parameters:
    - image_path: Path to the image containing the product.
    - model_id: The model ID used for inference (default is set to a Roboflow model for OCR).

    Returns:
    - List of detected expiry date texts and bounding boxes.
    """
    # Perform inference on the image
    result = CLIENT.infer(image_path, model_id=model_id)
    
    if 'predictions' not in result or len(result['predictions']) == 0:
        print("No predictions found.")
        return []

    # Load the image for cropping
    img = Image.open(image_path)

    detected_dates = []  # To store detected expiry dates and their bounding boxes

    # Loop through all predictions
    for idx, detection in enumerate(result['predictions']):
        # Step 1: Extract bounding box coordinates
        x, y, width, height = detection['x'], detection['y'], detection['width'], detection['height']

        # Step 2: Calculate bounding box edges
        left = x - width / 2
        top = y - height / 2
        right = x + width / 2
        bottom = y + height / 2

        # Step 3: Crop the detected region from the image
        cropped_img = img.crop((left, top, right, bottom))

        # Save the cropped image for optional verification (cropped images are saved with indices)
        cropped_img_path = f'cropped_expiry_date_{idx}.jpg'
        cropped_img.save(cropped_img_path)

        # Step 4: Perform OCR on the cropped image
        cropped_img = cropped_img.convert('L')  # Convert to grayscale for better OCR accuracy
        expiry_date_text = pytesseract.image_to_string(cropped_img, config='--psm 6')

        print(f"Detected Expiry Date Text for Detection {idx + 1}: {expiry_date_text}")

        # Step 5: Extract the date using a regular expression
        # Pattern: Supports MM/DD/YYYY or DD/MM/YYYY
        date_pattern = r'\b(0?[1-9]|1[0-2])[/\-](0?[1-9]|[12][0-9]|3[01])[/\-]((19|20)\d\d)\b'
        match = re.search(date_pattern, expiry_date_text)

        if match:
            expiry_date = match.group()
            print(f"Expiry Date for Detection {idx + 1}: {expiry_date}")
            detected_dates.append({"expiry_date": expiry_date, "bounding_box": (left, top, right, bottom)})
        else:
            print(f"No valid date found for Detection {idx + 1}.")
            detected_dates.append({"expiry_date": None, "bounding_box": (left, top, right, bottom)})

    return detected_dates


# # Example usage
# if __name__ == "__main__":
#     # Replace with your actual image path
#     image_path = "/content/sample_expiry.jpeg"

#     # Call the function and print the results
#     expiry_dates = detect_expiry_date(image_path)
#     for idx, date_info in enumerate(expiry_dates):
#         if date_info['expiry_date']:
#             print(f"Detection {idx + 1}: Expiry Date - {date_info['expiry_date']}")
#         else:
#             print(f"Detection {idx + 1}: No valid expiry date found.")