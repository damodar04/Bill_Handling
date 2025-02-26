import streamlit as st
import zipfile
import os
import io
import pandas as pd
import requests
import json
from google.cloud import vision
import re
import shutil

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


# Function to extract text from an image using Google OCR
def extract_text_from_image(image_path):
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    extracted_text = texts[0].description if texts else ""
    return extracted_text


# Function to process extracted text using Hugging Face LLM
def process_text_with_huggingface(text):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {
        "inputs": f"""Extract structured data from the provided bill text.

        Return **only** the JSON output. Do **not** include any explanations, additional text, formatting, or repetitions.

        Respond strictly in this JSON format:

        {{
          "Date": "",
          "Time": "",
          "Bill Type": "",
          "Store Name": "",
          "Total Amount": "",
          "Items": [
            {{"Item Name": "", "Qty": "", "Rate": "", "Amount": ""}}
          ]
        }}

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

        # Extract only the valid JSON part using regex
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            st.error("Invalid response format from API")
            return {}

    except json.JSONDecodeError as e:
        st.error(f"Error parsing response: {e}")
        return {}


# Function to extract images from zip file, including subfolders
def extract_images_from_zip(zip_file):
    extracted_files = []
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        # Traverse through all files in the zip
        for file_info in zip_ref.infolist():
            # Check if the file is an image and avoid extracting folder names
            if file_info.filename.lower().endswith((".png", ".jpg", ".jpeg")):
                extracted_files.append(file_info.filename)
                zip_ref.extract(file_info.filename, "extracted_bills")
    return extracted_files


# Streamlit UI
st.title("Expense Bill Extractor")

uploaded_file = st.file_uploader("Upload a zip file of scanned bills", type="zip")

if uploaded_file:
    st.write("Extracting ZIP file...")

    # Clear previous extraction
    extract_dir = "extracted_bills"
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            file_path = os.path.join(extract_dir, file)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)  # ✅ Delete folders
            else:
                os.remove(file_path)  # ✅ Delete files

    # Extract images from the zip file
    extracted_files = extract_images_from_zip(uploaded_file)
    
    if not extracted_files:
        st.warning("No image files found in the ZIP.")
    else:
        st.write("Extracted files:", extracted_files)

    extracted_data = []

    # Process each image in the extracted list
    for file in extracted_files:
        file_path = os.path.join(extract_dir, file)
        extracted_text = extract_text_from_image(file_path)
        if not extracted_text:
            st.error(f"Failed to extract text from {file}")
            continue

        structured_data = process_text_with_huggingface(extracted_text)

        if structured_data:
            extracted_data.append(structured_data)
        else:
            st.warning(f"Could not process {file}")

    # Create DataFrame and Export to Excel
    if extracted_data:
        df = pd.DataFrame(extracted_data)
        st.write(df)  # Debugging: Show extracted data
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

