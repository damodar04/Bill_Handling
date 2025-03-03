import streamlit as st
import zipfile
import os
import io
import fitz
import pandas as pd
import requests
import json
from google.cloud import vision
import re
import shutil

# Ensure 'static/' directory exists
if not os.path.exists("static"):
    os.makedirs("static")

# Set Google OCR Credentials
# Load credentials from Streamlit secrets
google_credentials = json.loads(st.secrets["google_cloud"]["credentials"])

# Save credentials to a temporary file
with open("google_key.json", "w") as f:
    json.dump(google_credentials, f)

# Set environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_key.json"
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\a\Desktop\IMP\google_key\textextract-1-09be363085ce.json"

# Initialize Google OCR Client
client = vision.ImageAnnotatorClient()

# Hugging Face API Key
HUGGINGFACE_API_KEY = "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"

# Define unwanted keywords to ignore instructional images
UNWANTED_KEYWORDS = ["instructions", "terms", "guidelines", "help", "support", "important"]


# Function to extract text from an image using Google OCR
def extract_text_from_image(image_path):
    if not os.path.exists(image_path):
        st.error(f"File not found: {image_path}")
        return ""

    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    extracted_text = texts[0].description if texts else ""

    # Skip images that contain unwanted instructional keywords
    if any(keyword in extracted_text.lower() for keyword in UNWANTED_KEYWORDS):
        st.warning(f"Ignoring {image_path} (Instructional Page)")
        return ""

    return extracted_text


# Function to process extracted text using Hugging Face LLM
def process_text_with_huggingface(text):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {
        "inputs": f"""Extract structured data from the provided bill text.

           Return **only** the JSON output. Do **not** include any explanations, additional text, formatting, or repetitions.

           Ensure that:
           - The "Time" field is strictly in the format **HH:MM** (12-hour format strictly) (without AM/PM). If the time is unavailable or incorrectly formatted, return an empty string "".
           - The "Time (AM/PM)" field should contain **only** "AM" or "PM" (no hours or minutes). If the time is unavailable, return an empty string "".        - The "Date" must be strictly formatted as **DD-MM-YYYY**, ensuring the month (MM) is always represented in **numbers (01-12)** and never in words. Ensure (YYYY) is always a full **four-digit year (2020)** and never in the format **(20)**. If the date is unavailable, leave this field blank ""—do not infer or approximate it.
           - The "Bill Type" must be one of the following categories: **"food", "flight", or "cab"**. If it does not fit into these categories, return an empty string "".
           - The "Bill Amount" should include the **currency symbol** (e.g., **$100, ₹500, €30**). If the currency is missing, do not infer it—return an empty string "".
           - The "Details" field should:
              - Contain the **restaurant name** if the "Bill Type" is **food**.
              - Contain the **"From - To"** locations if the "Bill Type" is **cab" or **flight**, inferred from terms like "From," "To," "Departure," "Arrival," or similar.
              - If this information is missing, return an empty string "".
           Respond strictly in this JSON format:

           {{
               "Date": "<DD-MM-YYYY or '' if unavailable>",
               "Time": "<HH:MM or '' if unavailable>",
               "Time (AM/PM)": "<AM/PM or '' if unavailable>",
               "Bill Type": "<food/flight/cab or '' if unavailable>",
               "Bill Amount": "<currency symbol + amount or '' if unavailable>",
               "Details": "<restaurant name or 'From - To' for flights/cabs or '' if unavailable>"
           }}

           Example Input:
           ```
           Bill: XYZ Restaurant  
           Date: January 5, 2024  
           Time: 15:45 PM  
           Type: Meal  
           Amount: 500 INR  
           ```

           Expected JSON Output:
           ```json
           {{
               "Date": "05-01-2024",
               "Time": "03:45",
               "Time (AM/PM)": "PM",
               "Bill Type": "food",
               "Bill Amount": "₹ 500",
               "Details": "XYZ Restaurant"
           }}
           ```

           Example Input:
           ```
           Bill: Uber Ride  
           Date: 12-02-2024  
           Time: 08:30 AM  
           Type: Cab Fare  
           Amount: $25  
           From: Downtown  
           To: Airport  
           ```

           Expected JSON Output:
           ```json
           {{
               "Date": "12-02-2024",
               "Time": "08:30",
               "Time (AM/PM)": "AM",
               "Bill Type": "cab",
               "Bill Amount": "$25",
               "Details": "Downtown - Airport"
           }}
           ```
           ### Bill Text:
           {text}

           ### JSON Output:
           """,
        "parameters": {"return_full_text": False}
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        st.error(f"API Error: {response.status_code} - {response.text}")
        return {}

    try:
        json_response = response.json()
        raw_text = json_response[0].get("generated_text", "").strip()
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            st.error("Invalid response format from API")
            return {}
    except json.JSONDecodeError as e:
        st.error(f"Error parsing response: {e}")
        return {}


# Function to extract images from ZIP file
# Function to extract images from ZIP file
def extract_images_from_zip(zip_file):
    extract_dir = "extracted_bills"
    extracted_images = []  # Store only image file paths
    extracted_pdfs = []  # Store PDF paths separately

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

        for file_info in zip_ref.infolist():
            file_path = os.path.join(extract_dir, file_info.filename)

            if file_info.filename.lower().endswith((".png", ".jpg", ".jpeg")):
                extracted_images.append(file_path)  # Store images
            elif file_info.filename.lower().endswith(".pdf"):
                extracted_images.extend(extract_images_from_pdf(file_path))  # Extract images directly

    return extracted_images, extracted_pdfs  # Return both separately


# Function to extract images from PDF (filters small images)


def extract_images_from_pdf(pdf_path, output_folder="extracted_images"):
    os.makedirs(output_folder, exist_ok=True)
    extracted_files = []

    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        images = page.get_images(full=True)
        for j, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = os.path.join(output_folder, f"page_{i + 1}_img_{j + 1}.{image_ext}")

            with open(image_filename, "wb") as f:
                f.write(image_bytes)

            extracted_files.append(image_filename)

    return extracted_files

# Streamlit UI
st.title("Expense Bill Extractor")

uploaded_file = st.file_uploader("Upload a zip file of scanned bills or a PDF", type=["zip", "pdf"])

if uploaded_file:
    st.write("Processing uploaded file...")

    extract_dir = "extracted_bills"
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)

    extracted_files = []

    if uploaded_file.name.endswith(".zip"):
        extracted_images, extracted_pdfs = extract_images_from_zip(uploaded_file)  # Separate images & PDFs
    else:  # Direct PDF upload
        extracted_pdfs = [os.path.join("extracted_bills", "uploaded.pdf")]
        with open(extracted_pdfs[0], "wb") as f:
            f.write(uploaded_file.getbuffer())
        extracted_images = []  # No direct images if a single PDF is uploaded

    # Process PDFs only once here (outside the zip function)
    for pdf in extracted_pdfs:
        extracted_images += extract_images_from_pdf(pdf)  # Convert PDF to images

    # Now extracted_images contains images from both direct uploads & PDFs
    if not extracted_images:
        st.warning("No valid images found in the file.")
    else:
        st.write("Extracted files:", extracted_images)

    # Process extracted images
    extracted_data = []
    for file in extracted_images:
        extracted_text = extract_text_from_image(file)
        if not extracted_text:
            continue  # Skip empty texts

        structured_data = process_text_with_huggingface(extracted_text)
        if structured_data:
            extracted_data.append(structured_data)

    if extracted_data:
        df = pd.DataFrame(extracted_data)
        df.drop_duplicates(inplace=True)  # Remove duplicate rows
        df.reset_index(drop=True, inplace=True)  # Reset index
        st.write(df)

        excel_file = io.BytesIO()
        with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False)
        excel_file.seek(0)

        st.download_button(
            label="Download Extracted Data as Excel",
            data=excel_file,
            file_name="expense_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.error("No structured data extracted.")

st.write("Upload a zip file containing scanned images of expense bills.")

