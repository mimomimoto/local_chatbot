from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import logging
import glob
import collections
import pickle
import os
import fitz
from PIL import Image
import pdfplumber
import openai
from io import BytesIO
import base64
from openai import OpenAI
import ollama
import streamlit as st
import sys


vectordb_root = "./db/"
pdf_list_root = "./pdf/"

API_KEY = ""

def extract_image(load_image_path, page_list):
    image_name_list = glob.glob(load_image_path + "/*")
    image_name_list = [image_name.replace(".png", "").split("/")[-1] for image_name in image_name_list]
    image_name_list = [int(image_name) for image_name in image_name_list]
    
    extract_image_list = list(set(image_name_list) & set(page_list))
    return extract_image_list

    
def start_localbot(model_name, load_name, user_msg):
    similarity_search_page_size = 5

    load_db_path = os.path.join(vectordb_root, load_name)
    load_vector_path = os.path.join(load_db_path,'vector')
    load_image_path = os.path.join(load_db_path,'img')
    load_txt_path = os.path.join(load_db_path,'txt')

    documents = pickle.load(open(f'{load_txt_path}/contents.txt', "rb"))
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    vectordb = FAISS.load_local(load_vector_path, embeddings, allow_dangerous_deserialization=True)
    
    answer = vectordb.similarity_search_with_score(user_msg, k = similarity_search_page_size)

    page_list = []
    
    for i in answer:
        answer, score = i
        page_list.append(answer.metadata["page"])

    page_list_collection = collections.Counter(page_list)
    page_list = set(page_list)
    ref_p_num = len(page_list)
    context = ""
    for i in range(ref_p_num):
        tmp = documents[page_list_collection.most_common()[i][0]].page_content
        tmp = tmp.replace("\n", "")
        context += tmp + "\n"
    
    extract_image_list = extract_image(load_image_path, page_list)

    for extract_image_name in extract_image_list:
        print(f"Page {extract_image_name} contains image")
    
    response = ollama.chat(model=model_name, messages=[
        {
            'role': 'system',
            'content': 'あなたはユーザの質問に対して、与えられたコンテキスト情報をもとに回答を丁寧に生成するモデルです。',
        },
        {
            'role': 'user',
            'content': f'{user_msg}\n\n{context}',
        },
    ])
    
    chat_response = response['message']['content']
    
    return chat_response, extract_image_list
        
def start_st_chat_bot(model_name, sp_name):
    load_db_path = os.path.join(vectordb_root, sp_name)
    load_image_path = os.path.join(load_db_path, 'img')

    st.title("高知市地域防災チャットボット")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.chat_message("assistant").markdown("質問はありますか？")
    user_msg = st.chat_input("質問を入力してください")

    if user_msg:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        response, image_path_list = start_localbot(model_name, sp_name, user_msg)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response, "images": image_path_list})

        # Display conversation history
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.chat_message(chat["role"]).markdown(chat["content"])
            else:
                assistant_message = st.chat_message(chat["role"])
                assistant_message.markdown(chat["content"])
                
                # Display images if present
                if "images" in chat and chat["images"]:
                    for image_path in chat["images"]:
                        save_image_path = os.path.join(load_image_path, f'{image_path}.png')
                        assistant_message.image(save_image_path, caption=f"Page {image_path}")


def main():
    args = sys.argv[1:]
    
    model_name = args[0]
    pdf_path = args[1]
    sp_name = pdf_path.split("/")[-1].split(".")[0]
    
    start_st_chat_bot(model_name, sp_name)
    

if __name__ == "__main__":
    main()