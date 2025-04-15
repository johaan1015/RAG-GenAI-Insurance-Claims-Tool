import os
import tempfile
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


os.environ["OPENAI_API_KEY"] = "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"
def display_document_context(documents):
 with st.expander("Context Snippets"):
    for doc in documents:
        page_content = doc.page_content.strip().replace("$", "\\$")
        source = os.path.basename(doc.metadata["source"])
        page_number = doc.metadata.get('page', 'Unknown Page')
        html_content = f"""
        <blockquote style="border-left: 2px solid #d0d0d0; padding-left: 10px; margin-left: 0; margin-right: 0; color: #333;">
            {page_content}
            <footer style="text-align: right; font-size: 0.9em; color: #555;">
                <strong>Source:</strong> {source} <br>
                <strong>Page:</strong> {page_number}
            </footer>
        </blockquote>
        """
        st.markdown(html_content, unsafe_allow_html=True)      
        
def run_backend_code(pdf_files,user_question):
    documents = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for pdf_file in pdf_files:
            pdf_path = os.path.join(temp_dir, pdf_file.name)
            with open(pdf_path, 'wb') as f:
                f.write(pdf_file.getbuffer())
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunked_documents = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.from_documents(chunked_documents,embeddings)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    retriever_from_llm = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=llm)
    
    prompt=prompt = ChatPromptTemplate.from_template("""
            Answer the following question based only on the provided context. 
            Think step by step before providing a detailed answer. 
            I will tip you $1000 if the user finds the answer helpful. 
            <context>
            {context}
            </context>
            Question: {input}""")
   
    
    document_chain=create_stuff_documents_chain(llm,prompt)
    

    retrieval_chain=create_retrieval_chain(retriever_from_llm,document_chain)
    response=retrieval_chain.invoke({"input":user_question})
    
    return response
   
def show():
    st.title('Question Answering System')

    st.markdown("""
        <style>
        /* General styling for file uploader */
        .stFileUploader {
            border: 2px dashed var(--primary-color);
            padding: 20px;
            border-radius: 10px;
            background-color: var(--background-color);
        }
        /* Button styling */
        .stButton button {
            background-color: var(--button-bg-color);
            color: var(--button-text-color);
            border: none;
            padding: 10px 20px;
            text-align: center;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 5px;
        }
        /* Spinner styling */
        .stSpinner {
            font-size: 20px;
            font-weight: bold;
            color: var(--spinner-color);
        }
        /* Result box styling */
        .result-box {
            border: 1px solid var(--border-color);
            padding: 20px;
            border-radius: 10px;
            background-color: var(--box-bg-color);
            color: var(--text-color);
        }

        /* Define colors that adapt to light and dark modes */
        :root {
            --primary-color: #4CAF50;
            --background-color: #f9f9f9;
            --button-bg-color: #4CAF50;
            --button-text-color: white;
            --spinner-color: #4CAF50;
            --border-color: #4CAF50;
            --box-bg-color: #ffffff;
            --text-color: #000000;
        }
        [data-theme='dark'] {
            --primary-color: #81C784;
            --background-color: #333333;
            --button-bg-color: #81C784;
            --button-text-color: black;
            --spinner-color: #81C784;
            --border-color: #81C784;
            --box-bg-color: #424242;
            --text-color: #ffffff;
        }
        </style>
    """, unsafe_allow_html=True)
    if "messages" not in st.session_state:
        st.session_state.messages=[]

    for message in st.session_state.messages:
     with st.chat_message(message["role"]):
        st.markdown(message["content"])

    if user_question :=st.chat_input("Ask a question about the claim application"):
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role":"user","content":user_question})
        if 'uploaded_files' in st.session_state:
            uploaded_files = st.session_state['uploaded_files']
            with st.spinner('Analyzing documents...'):
                result= run_backend_code(uploaded_files,user_question)
            final_result = result["answer"].replace("$", "\\$")
            with st.chat_message("assistant"):
             st.markdown(final_result)
            st.session_state.messages.append({"role":"assistant","content":final_result})

            display_document_context(result["context"])

        else:
            st.warning("No files uploaded. Please upload documents.")
