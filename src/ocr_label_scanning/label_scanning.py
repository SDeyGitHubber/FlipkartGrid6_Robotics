import requests

# Define your OCR.Space API key
API_KEY = 'K83950318488957'
OCR_SPACE_URL = 'https://api.ocr.space/parse/image'

def extract_text_from_image(image_path):
    with open(image_path, 'rb') as image_file:
        # Prepare the payload for the POST request
        payload = {
            'apikey': API_KEY,
            'language': 'eng',  # Specify the language
            'isOverlayRequired': True,
        }
        files = {
            'file': image_file
        }

        # Send POST request to OCR.Space
        response = requests.post(OCR_SPACE_URL, data=payload, files=files)

        # Check if the request was successful
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception("Error: " + str(response.status_code) + " - " + response.text)

def process_ocr_response(response):
    text_blocks = []
    if 'ParsedResults' in response:
        for result in response['ParsedResults']:
            text_blocks.append(result['ParsedText'])

    return text_blocks

if __name__ == "__main__":
    # Specify the path to your image
    image_path = '/content/Screenshot_ocr3.png'  # Change to your actual image file path

    try:
        # Extract text from the image
        response = extract_text_from_image(image_path)

        # Process and print the extracted text
        extracted_text = process_ocr_response(response)
        print("Extracted Text:")
        for text in extracted_text:
            print(text)
    except Exception as e:
        print(str(e))