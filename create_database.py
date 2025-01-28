from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import logging
import pickle
import os
import fitz
from PIL import Image
import pdfplumber
from io import BytesIO
import base64
from openai import OpenAI
import sys

vectordb_root = "./db/"
pdf_list_root = "./pdf/"

API_KEY = ""

def detect_table_with_pdfplumber(page_number, pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        if page_number < 1 or page_number > len(pdf.pages):
            return False
        page = pdf.pages[page_number]
        tables = page.extract_tables()
        return bool(tables)
    
def encode_image_pil(image):
    # Convert a PIL image to a base64-encoded string
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def save_db(path, save_name):
    save_db_path = os.path.join(vectordb_root, save_name)
    save_vector_path = os.path.join(save_db_path,'vector')
    save_image_path = os.path.join(save_db_path,'img')
    save_txt_path = os.path.join(save_db_path,'txt')
    
    os.makedirs(f'{save_vector_path}', exist_ok=True)
    os.makedirs(f'{save_image_path}', exist_ok=True)
    os.makedirs(f'{save_txt_path}', exist_ok=True)
    
    pdf_document = fitz.open(path)
    lang_chain_pdf_document = PyPDFLoader(path).load()
    zoom = 2.5
    
    client = OpenAI(api_key=API_KEY)

    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        image_list = page.get_images(full=True)
        has_table = detect_table_with_pdfplumber(page_number, path)
        
        if image_list or has_table:
            print(f"Page {page_number} contains {len(image_list)} image(s) or table(s). Converting the page to an image...")
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            image_filename = f"{save_image_path}/{page_number}.png"
            image.save(image_filename)
            print(f"Page {page_number} converted and saved as {image_filename}")
            
            base64_image = encode_image_pil(image)

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "この画像に記載されている情報を詳細に教えてください"},
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                        },
                        },
                    ],
                    }
                ],
                max_tokens=1000,
            )
            lang_chain_pdf_document[page_number].page_content = response.choices[0].message.content
            

        else:
            print(f"Page {page_number + 1} does not contain images or tables. Skipping conversion.")

    
    f = open(f'{save_txt_path}/contents.txt', "wb")
    pickle.dump(lang_chain_pdf_document, f)

    text_splitter_1 = CharacterTextSplitter(chunk_size=10, chunk_overlap=0, separator = "。")
    text_splitter_2 = CharacterTextSplitter(chunk_size=10, chunk_overlap=0, separator = "\n")
    logger = logging.getLogger("langchain")
    logger.setLevel(logging.ERROR)
    texts_1 = text_splitter_1.split_documents(lang_chain_pdf_document)
    texts_2 = text_splitter_2.split_documents(lang_chain_pdf_document)
    texts = texts_1 + texts_2
    for i in range(len(texts)):
        texts[i].page_content = texts[i].page_content.replace("\n", "")

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    vectordb = FAISS.from_documents(texts, embeddings)

    vectordb.save_local(save_vector_path)
    
    print("SUCCESS: Vectordb saved")

def main():
    args = sys.argv[1:]
    pdf_path = args[0]
    sp_name = pdf_path.split("/")[-1].split(".")[0]
    save_db(pdf_path, sp_name)

if __name__ == "__main__":
    main()