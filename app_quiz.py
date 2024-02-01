import os
from key import openai_api_key
import streamlit as st
from PyPDF2 import PdfReader  # Reads pdf file.
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

os.environ['OPENAI_API_KEY']= openai_api_key

st.set_page_config(page_title="Generate My Quiz")
st.header("💬Generate Quiz from my Notes🗒️")

#upload file
pdf= st.file_uploader("Upload your pdf", type="pdf")

#extracting the text from pdf
if pdf is not None:
    pdf_reader= PdfReader(pdf)
    text=""
    for page in pdf_reader.pages:
        text +=page.extract_text()


    
    prompt= PromptTemplate(input_variables=[text],
                             template="""
                             You are a Quiz Generator that can generate mixed short answer questions, Multiple Choice questions about the lecture notes videos based on the pdf uploaded
                             Generate 10 Quiz Questions based only on the pdf data.
                             By using data from the following pdf : {text}
                             The Quiz questions should be 10 questions in total with a mixture of short answer questions, 
                             Multiple Choice questions with one correct answers and 3 wrong answers. Display options on new line after each question.
                             Mention the correct answer, like Answer: Correct Answer on new Line
                             Format the questions correctly
                             If you feel like you dont have enough information to answer the question, say "I dont know".
                             """,
                            )
    
    llm= OpenAI()
    chain= LLMChain(llm=llm, prompt=prompt) 
    response= chain.run(text=text)
    st.write(response)

    
