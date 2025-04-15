import os
import tempfile
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate

os.environ["OPENAI_API_KEY"] = "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"

def get_content(chunked_documents,llm):
    map_prompt = """
    Write a concise summary and extract key data points from the following:
    "{text}"
    KEY DATA POINTS:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    combine_prompt = """
    Extract key data points from the following text delimited by triple backquotes and then provide a concise summary.
    Return your response in two sections: Key Data Points and Concise Summary.
    ```{text}```
    KEY DATA POINTS:
    - 

    CONCISE SUMMARY:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
    summary_chain = load_summarize_chain(llm=llm,
                                     chain_type='map_reduce',
                                     map_prompt=map_prompt_template,
                                     combine_prompt=combine_prompt_template,
                                    )
    output=summary_chain.run(chunked_documents)

    return output

def run_backend_code(pdf_files):
    with tempfile.TemporaryDirectory() as temp_dir:
        for pdf_file in pdf_files:
            documents = []
            pdf_path = os.path.join(temp_dir, pdf_file.name)
            with open(pdf_path, 'wb') as f:
                f.write(pdf_file.getbuffer())
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunked_documents = text_splitter.split_documents(documents)
            vectordb = FAISS.from_documents(
            documents=chunked_documents,
            embedding=OpenAIEmbeddings(),
            )
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
            retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectordb.as_retriever(), llm=llm)
    
            query = "Classify the Document"
            

            relevant_docs = retriever_from_llm.get_relevant_documents(query)
            
            classification_prompt_template = """
            Based on the following context, classify the type of insurance document. Return the answer from among the following classes:-
            Claim Form
            Policy Coverage Document
            Police Report
            Repair Estimate Bills
            Medical Bills
            First Notice of Loss(FNOL)
            Vehicle Ownership Document
            Traffic Rules
            Context:
            {documents}

            Document Type:
            """
            
            prompt = ChatPromptTemplate.from_template(classification_prompt_template)
            
            rag_chain= prompt | llm
            answer = rag_chain.invoke({"documents": relevant_docs})
            
            with st.expander(answer.content):
                content = get_content(chunked_documents,llm)
                final_result = content.replace("$", "\\$")
                st.markdown(final_result, unsafe_allow_html=True)

def show():
    st.title('Summary of Documents')

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



    if st.button('Provide Summary of Documents'):
        if 'uploaded_files' in st.session_state:
            uploaded_files = st.session_state['uploaded_files']
            with st.spinner('Analyzing documents...'):
                run_backend_code(uploaded_files)
        else:
            st.warning("No files uploaded. Please upload documents.")
