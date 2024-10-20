import google.generativeai as genai

# Step 1: Configure the API Key and Model
genai.configure(api_key="AIzaSyBW8UrfLMN3blHpSElVlbDFuzleecYMDnA")  # Replace with your API key

# Step 2: Define the configuration for the model
generation_config = {
    "temperature": 0,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 4096,
}

# Step 3: Define the safety settings for the model
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    }
]

# Step 4: Load the Gemini Pro model with safety settings and generation config
model = genai.GenerativeModel(
    model_name="gemini-pro",
    safety_settings=safety_settings,
    generation_config=generation_config
)

# Step 5: Define a function to analyze the extracted text for product information
def analyze_text_with_gemini(extracted_text):
    prompt = f"""
    Analyze the following text and identify the product information including:
    - Product Name
    - Product Description (NOTE: Try to identify what product it is from the description in the text)
    - Serial Number (EAN)
    - Label
    - MRP (Maximum Retail Price)
    In case if some other information is present in the extracted text, then do add it in the
    product information such as:
    - Brand Name: The name of the brand or manufacturer.
    - Ingredients: List of ingredients used in the product.
    - Nutritional Information: Details about the nutritional content (for food products).
    - Usage Instructions: How to use the product.
    - Warnings: Safety warnings or precautions.
    - Manufacturing Date: Date when the product was manufactured.
    - Expiry Date: Date when the product expires.
    - Batch Number: Identifier for the batch of products.
    - Country of Origin: Where the product was made.
    - Certifications: Any certifications or quality marks.
    - Contact Information: Manufacturer’s contact details.
    - Barcode/QR Code: For scanning and additional information.
    - Storage Instructions: How to store the product.
    - Recycling Information: Instructions on how to recycle the packaging.
    - Calories: Nutritional information for food products.
    
    Whatever information you can get that resembles any of the above categories, you are
    required to output it such that the unavailable categories are not listed in the output.

    Text: {extracted_text}
    """

    # Step 6: Use the model to generate a response
    response = model.generate_content(prompt)

    # Step 7: Return the generated output
    return response.text

# Example Usage:
if __name__ == "__main__":
    # Step 8: Example of extracted text (replace with actual extracted text from OCR)
    extracted_text = """
    Product: XYZ Phone
    Description: A high-end smartphone with excellent features.
    EAN: 1234567890123
    Label: Electronics
    MRP: $799
    """
    
    # Step 9: Call the function to analyze the text
    analysis_result = analyze_text_with_gemini(extracted_text)

    # Step 10: Print the analysis result
    print("Analysis Result:")
    print(analysis_result)