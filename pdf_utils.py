import os
import requests
from pypdf import PdfReader
from config import DATA_PATH

def download_pdf(url):
    os.makedirs(DATA_PATH, exist_ok=True)
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download PDF")

    file_path = os.path.join(DATA_PATH, "paper.pdf")
    with open(file_path, "wb") as f:
        f.write(response.content)

    return file_path

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text