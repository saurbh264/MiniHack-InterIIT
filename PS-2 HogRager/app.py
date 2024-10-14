import os
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import CTransformers
import pickle
import faiss
from langchain.vectorstores import FAISS
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import warnings
warnings.filterwarnings("ignore")
import pickle

# drive.mount('/content/drive'
from langchain_community.document_loaders import JSONLoader
import json
from pathlib import Path
from pprint import pprint
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from hograger2 import (
    read_json_file,
    process_json_to_documents,
    split_documents, 
    create_embeddings_with_chroma,  
    format_output
)
import streamlit as st
import chromadb
# from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
# from pyngrok import ngrok


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
file_path='corpus.json'
json_data = json.loads(Path(file_path).read_text())

embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "gpu"}
@st.cache_resource


# def initialize_database():

#     chromadb.api.client.SharedSystemClient.clear_system_cache()
#     db_path = '/content/my_database.db'
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS collections (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             name TEXT NOT NULL
#         )
#     ''')
#     conn.commit()
#     conn.close()
  
def load_vector_store():
    db = "./db"
    file_path = 'corpus.json'  # Adjust path as needed
    json_data = read_json_file(file_path)
    docs = process_json_to_documents(json_data)
    split_docs = split_documents(docs)
    vector_store = create_embeddings_with_chroma(split_docs,embedding_model_name,model_kwargs,db)
    return vector_store






vector_store = load_vector_store()





# vector embedding generated



# setting up the llm to be used
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",  # Change to "flan-t5-large", "flan-t5-xl", or "flan-t5-xxl" as needed
    model_kwargs={"temperature": 0.55, "max_length": 512}
)

# Set up the conversational retrieval chain using imported functions
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vector_store.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True
)

if 'input_query' not in st.session_state:
    st.session_state.input_query = ""
# Initialize chat history
chat_history = []

# begining the chatbot
st.title("RAG Chatbot")
st.write("Ask me anything, and I'll retrieve relevant information!")

# Streamlit input box for user query
query = st.text_input("Enter your query:", key = "input_query", value=st.session_state.input_query)

# Run the chatbot logic if query is provided
if st.button("Generate Response"):
    # Process query using the QA chain
    if query is not None:
        if st.button("Clear"):
          st.session_state.input_query = query

        result = qa_chain({'question': query, 'chat_history': chat_history})
        
        # Format the output
        structured_output = format_output(result)
        
        # Display the chatbot response
        st.subheader("Answer:")
        st.write(structured_output['answer'])

        # Display evidence from source documents
        st.subheader("Evidence from the documents:")
        for evidence in structured_output['evidence_list']:
            st.write(f"**Title:** {evidence['title']}")
            st.write(f"**Author:** {evidence['author']}")
            st.write(f"**Source:** {evidence['source']}")
            st.write(f"**Category:** {evidence['category']}")
            st.write(f"**Published at:** {evidence['published_at']}")
            st.write(f"**URL:** {evidence['url']}")
            st.write(f"**Fact:** {evidence['fact'][:150]}...")  # Limiting the fact length for readability
        
        # Update chat history
        chat_history.append((query, structured_output['answer']))
        
    else:
        st.warning("Please upload an image and enter a question.")














