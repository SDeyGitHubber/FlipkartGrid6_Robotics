import streamlit as st
from datetime import datetime
from PIL import Image
import cv2
import os
from src.freshness_detection.preprocessing import preprocess_image
from src.freshness_detection.predictor import load_freshness_model, predict_freshness
from src.ocr_label_scanning.gemini_analysis import analyze_text_with_gemini
from src.ocr_expiry.expiry_date_detection import detect_expiry_date
from src.object_detection.detection import detect_and_count_objects
from src.ocr_label_scanning.label_scanning import extract_text_from_image, process_ocr_response  # Import OCR functions
from ultralytics import YOLO
from dotenv import load_dotenv
import shutil

# Load environment variables for API keys (if using .env file)
load_dotenv()

# Helper to save uploaded file
def save_uploaded_file(uploaded_file, folder='static/'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return file_path


# --- Main Page Setup ---
st.title("Product Analysis System")

# Select task
option = st.selectbox(
    "Choose a task:",
    ["Select...", "Complete OCR Analysis", "Object Detection And Count", "Freshness Detection"]
)

# --- Page Navigation ---
if option == "Complete OCR Analysis":
    st.header("Complete OCR Analysis")

    # Upload product image for OCR
    uploaded_image = st.file_uploader("Upload a product image", type=["png", "jpg", "jpeg"])

    if uploaded_image:
        image_path = save_uploaded_file(uploaded_image)
        
        # Display the uploaded image
        st.image(image_path, caption="Uploaded Product Image", use_column_width=True)

        # OCR options
        ocr_option = st.radio("Choose OCR Option:", ('Scanned OCR Details',))  # Removed "Get Expiry Date"

        if ocr_option == 'Scanned OCR Details':
            # Perform Gemini Pro analysis on OCR text
            st.write("Performing OCR analysis...")

            # Extract text from the uploaded image using OCR
            try:
                ocr_response = extract_text_from_image(image_path)  # Call the OCR extraction function
                extracted_text = process_ocr_response(ocr_response)  # Process the response to get the text
                
                # Combine all extracted text blocks into one string for analysis
                extracted_text_str = "\n".join(extracted_text)
                
                # Call the Gemini Pro analysis function with the extracted text
                analysis_result = analyze_text_with_gemini(extracted_text_str)
                
                # Display the OCR and Gemini Pro results
                st.write("OCR Analysis Result:")
                st.write(analysis_result)

                # # --- Expiry Date Detection Logic ---
                # expiry_date = None
                # try:
                #     detections = detect_expiry_date(image_path, "flipkartgrid-ocr-tmd6q-lbde9/1")
                #     for detection in detections:
                #         expiry_date = detection.get('expiry_date')
                #         if expiry_date:
                #             break

                # except Exception as e:
                #     st.error(f"Error detecting expiry date: {e}")

                # if expiry_date:
                #     st.success(f"Detected Expiry Date: {expiry_date}")
                # else:
                #     st.warning("Expiry date not found.")
            
            except Exception as e:
                st.error(f"Error performing OCR analysis: {e}")

if option == "Object Detection And Count":
    st.header("Object Detection And Count")
    
    # Debug message to check if this block is being run
    st.write("You are in the Object Detection section!")

    # Upload image for object detection
    uploaded_detection_image = st.file_uploader("Upload an image with multiple objects", type=["png", "jpg", "jpeg"])

    if uploaded_detection_image is not None:
        st.write(f"File Uploaded: {uploaded_detection_image.name}")  # Debug message to confirm file upload
        detection_image_path = save_uploaded_file(uploaded_detection_image)

        # Display the uploaded image
        st.image(detection_image_path, caption="Uploaded Image", use_column_width=True)

        # Perform object detection using YOLOv8
        st.write("Processing image for object detection...")

        try:
            # Load YOLOv8 model (ensure you have the correct model path)
            model_path = 'yolov8n.pt'
            model = YOLO(model_path)

            # Call detection function
            output_image_path = os.path.join('static', f"output_{uploaded_detection_image.name}")
            detected_objects, object_counts = detect_and_count_objects(detection_image_path, model, output_image_path)

            # Display number of objects and their descriptions
            st.write(f"Detected {sum(object_counts.values())} objects:")
            for obj, count in object_counts.items():
                st.write(f"- {count} {obj}(s)")

            st.write("Detected Objects and Descriptions:")
            for obj in detected_objects:
                st.write(f"Bounding Box: {obj['bounding_box']}, Shape: {obj['shape']}, Color: {obj['color']}, "
                         f"Label: {obj['label']}, Confidence: {obj['confidence']}")

            # Display the original image with bounding boxes
            st.image(output_image_path, caption="Detected Objects", use_column_width=True)

        except Exception as e:
            st.error(f"Error in object detection: {e}")

elif option == "Freshness Detection":
    st.header("Freshness Detection")

    # Upload image for freshness detection
    uploaded_freshness_image = st.file_uploader("Upload an image of a fruit or vegetable", type=["png", "jpg", "jpeg"])

    if uploaded_freshness_image:
        # Save the uploaded image
        freshness_image_path = save_uploaded_file(uploaded_freshness_image)
        
        # Display the uploaded image
        st.image(freshness_image_path, caption="Uploaded Image", use_column_width=True)
        
        # Load the pre-trained freshness detection model
        freshness_model = load_freshness_model()

        # Perform freshness detection using the loaded model and preprocessor
        predicted_class, prediction = predict_freshness(freshness_image_path, freshness_model, preprocess_image)

        # Display freshness prediction with bold text
        freshness_labels = ['fresh_apple', 'fresh_banana', 'fresh_bitter_gourd', 'fresh_capsicum',
                            'fresh_orange', 'fresh_tomato', 'stale_apple', 'stale_banana', 'stale_bitter_gourd',
                            'stale_capsicum', 'stale_orange', 'stale_tomato']  # Example labels, adjust as needed
        freshness_status = freshness_labels[predicted_class]
        
        # Display the status in bold using Markdown
        st.markdown(f"**Freshness Status: {freshness_status}**")

# Cleanup uploaded files after processing to free up space
def clean_up(folder='static/'):
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            st.error(f"Error deleting file {file_path}: {e}")

# Add a button to manually clean up the static folder
if st.button('Clean Up Uploaded Files'):
    clean_up()
    st.success("Temporary files cleaned up!")