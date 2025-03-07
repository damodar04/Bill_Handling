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
from streamlit import spinner

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


# Function to process extracted text using Hugging Face LLM with refined prompt
def process_text_with_huggingface(text):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {
        "inputs": f"""Extract structured data from the given bill text with maximum accuracy. 

            Return **only** JSON output without explanations. 

            ### Rules:
            - **Date**: Format **DD-MM-YYYY** (e.g., 05-01-2024). Convert formats like DD/MM/YYYY, YYYY-MM-DD, and DD Mon YYYY.
            - **Time**: Format **HH:MM** (12-hour, e.g., 03:45).**Exclude AM/PM**.
            - **Time (AM/PM)**: Extract **only** "AM" or "PM", else "".
            - **Bill Type**: Categorize as **"food"**, **"flight"**, or **"cab"** based on keywords.
            - "Currency Name": Extract currency code (e.g., USD, INR, EUR) or infer from symbols (e.g., $ → USD, ₹ → INR). 
              - should **not include** any other number or alphabet other than currency symbol
              - If unavailable, return "".
           - "Bill Amount": Extract as **<currency symbol><amount>** (e.g., $25, ₹500). 
              - Include symbol if present; otherwise, return numeric amount only (e.g., 25). 
              - **Do not include any other characters, numbers, or alphabets.**
              - Convert codes like "INR" → "₹", "USD" → "$", "EUR" → "€".  
              - If the symbol is missing or unrecognized, return "".
            - **Details**:
              - "food": **only** Extract restaurant name.
              - "flight"/"cab": Extract **only "From: <location> - To: <location>"**.
              - If missing, return "".

            Examples:
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
               "Date": "05/01/2024",
               "Time": "03:45",
               "Time (AM/PM)": "PM",
               "Bill Type": "food",
               "Currency Name": "INR",
               "Bill Amount": "₹ 500",
               "Details": "XYZ Restaurant"
           }}
           Example Input:
           ```
           Bill: Airline Ticket
           Date: 2024-03-10
           Time: 22:15
           Amount: €120
           Departure: London Heathrow
           Arrival: Paris CDG
           '''
           Expected JSON Output:
           ```json
           {{
            "Date": "10/03/2024",
            "Time": "10:15",
            "Time (AM/PM)": "PM",
            "Bill Type": "flight",
            "Currency Name": "EUR",
            "Bill Amount": "€120",
            "Details": "From: London Heathrow - To: Paris CDG"
            }}
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
               "Date": "12/02/2024",
               "Time": "08:30",
               "Time (AM/PM)": "AM",
               "Bill Type": "cab",
               "Currency Name": "USD",
               "Bill Amount": "$25",
               "Details": "From: Downtown - To: Airport"
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
# Custom CSS for background and text color
# Custom CSS for background and text color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFF5E1 !important;  /* Cream background */
        color: black !important;
    }
    .stSidebar {
        background-color: #333 !important; /* Dark gray sidebar */
        color: white !important;
    }
    .stButton>button {
        background-color: #ff5733 !important; /* Orange button */
        color: white !important;
    }
    .stDataFrame table {
        background-color: rgba(255, 245, 225, 0) !important; /* Fully transparent background */
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Expense Bill Extractor")

# Sidebar with icons
st.sidebar.title("Upload Section")
st.sidebar.image("https://img.icons8.com/ios-filled/50/000000/upload.png", width=50)
uploaded_file = st.sidebar.file_uploader(
    "Upload a zip file of scanned bills or a PDF",
    type=["zip", "pdf"],
    key="unique_file_uploader"
)

st.sidebar.title("Categories")
st.sidebar.image("https://img.icons8.com/ios-filled/50/000000/restaurant.png", width=100)
st.sidebar.write("Food Bills")
st.sidebar.image("https://img.icons8.com/ios-filled/50/000000/airport.png", width=100)
st.sidebar.write("Flight Bills")
st.sidebar.image("https://img.icons8.com/ios-filled/50/000000/taxi.png", width=100)
st.sidebar.write("Cab Bills")

# uploaded_file = st.sidebar.file_uploader(
#     "Upload a zip file of scanned bills or a PDF",
#     type=["zip", "pdf"],
#     key="unique_file_uploader"
# )

# Variable to store extracted filenames
extracted_filenames = []
extracted_images = []
extracted_pdfs = []

if uploaded_file:
    st.sidebar.success("File uploaded successfully! ✅")

    # Create an extraction directory
    extract_dir = "extracted_bills"
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)

    # Extract files but don't process yet
    if uploaded_file.name.endswith(".zip"):
        extracted_images, extracted_pdfs = extract_images_from_zip(uploaded_file)
    else:
        extracted_pdfs = [os.path.join(extract_dir, "uploaded.pdf")]
        with open(extracted_pdfs[0], "wb") as f:
            f.write(uploaded_file.getbuffer())

    # Extract images from PDFs
    for pdf in extracted_pdfs:
        extracted_images += extract_images_from_pdf(pdf)

    if not extracted_images:
        st.warning("No valid images found in the file.")
    else:
        extracted_filenames = [os.path.basename(file) for file in extracted_images]
        st.write("Extracted Files:")
        st.write(extracted_filenames)  # Show filenames

    # Show the "Process" button
    if extracted_filenames:
        if st.button("Process Extracted Files"):
            with spinner("Processing extracted files..."):
                extracted_data = []
                for file in extracted_images:
                    extracted_text = extract_text_from_image(file)
                    if not extracted_text:
                        continue  # Skip empty texts

                    structured_data = process_text_with_huggingface(extracted_text)
                    if structured_data:

                        # Separate currency from Bill Amount
                        bill_amount = structured_data.get("Bill Amount", "")
                        currency = bill_amount[0] if bill_amount else ""
                        structured_data["Bill Amount"] = bill_amount[1:].strip() if bill_amount else ""

                        # Add currency name dynamically
                        structured_data["Currency Name"] = currency

                        # Format "From - To" for flight or cab bills
                        if structured_data["Bill Type"] in ["flight", "cab"]:
                            details = structured_data.get("Details", "")
                            if "from" in details.lower() and "to" in details.lower():
                                from_to_match = re.search(r"from\s*:\s*(.*?)\s*-\s*to\s*:\s*(.*)", details,
                                                          re.IGNORECASE)
                                if from_to_match:
                                    structured_data[
                                        "Details"] = f"From: {from_to_match.group(1).strip()} - To: {from_to_match.group(2).strip()}"

                        extracted_data.append(structured_data)

            # Display results
            if extracted_data:
                df = pd.DataFrame(extracted_data)
                df.drop_duplicates(inplace=True)
                df.reset_index(drop=True, inplace=True)
                st.write(df)

                # Prepare downloadable Excel file
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
