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
import PyPDF2
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


os.environ["OPENAI_API_KEY"] = "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"
def extract_text_from_pdf(pdf_file):
            text = ""
            reader = PyPDF2.PdfReader(pdf_file)    
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
            return text

def classify_document(text):
            classification_prompt_template = """
            Based on the following context, classify the type of insurance document. Return the answer from among the following classes.:-
            Claim Form
            Policy Coverage Document
            Police Report
            Repair Estimate Bills
            Medical Bills
            First Notice of Loss(FNOL)
            Vehicle Ownership Document
            Traffic Rules

            Context:
            {context}

            Document Type:
            """
            
            prompt = ChatPromptTemplate.from_template(classification_prompt_template)
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0
            )
            rag_chain= prompt | llm
            answer = rag_chain.invoke({"context": text})
            return answer.content

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

def run_backend_code(pdf_files):
        pdf_of_interest=[]
        for pdf_file in pdf_files:
            text=extract_text_from_pdf(pdf_file)
            type_of_document=classify_document(text)
            print(type_of_document)
            if(type_of_document=="Policy Coverage Document" or type_of_document=="Repair Estimate Bills" or type_of_document=="Medical Bills"):
                 pdf_of_interest.append(pdf_file)
        documents=[]
        with tempfile.TemporaryDirectory() as temp_dir:
            for pdf_file in pdf_of_interest:
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
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 8})
        
        query = """You are an insurance claims analyst.Based on the context, please answer the following questions with proper justification:
        1. What are the key coverage details provided in the policy, including maximum coverage limits?
        2. What are the total costs listed in the repair estimate bills, medical bills, and other costs of rendered services?
        3. Does the policy cover all the costs listed in the repair estimate bills, medical bills, and other rendered services? Please provide a detailed comparison.
        4. What is the maximum coverage provided by the policy for each type of service (repair, medical, etc.)?
        5. Does the total cost of the rendered services exceed the maximum coverage provided by the policy?
        6. What is your decision regarding whether the policy covers all the rendered services? Provide a detailed justification for your decision.
        """
        
        chat_prompt_template = """
        Answer the following question based only on the provided context.
        <context>
        {context}
        </context>
        Question: {input}
        """
        
        prompt = ChatPromptTemplate.from_template(chat_prompt_template)
        
        document_chain=create_stuff_documents_chain(llm,prompt)
        

        retrieval_chain=create_retrieval_chain(retriever,document_chain)
        response=retrieval_chain.invoke({"input":query})
    
        return response
def show():
    st.title('Eligibiity Check')
    st.write("Assess coverage eligibility by comparing policy details with service invoices to determine if the services are covered and identify the maximum allowable coverage")

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



    if st.button('Conduct Eligibility Check'):
        if 'uploaded_files' in st.session_state:
            uploaded_files = st.session_state['uploaded_files']
            with st.spinner('Analyzing documents...'):
             result= run_backend_code(uploaded_files)
             escaped_result = result["answer"].replace("$", "\\$")
            
            st.markdown(escaped_result)
            display_document_context(result["context"])
        else:
            st.warning("No files uploaded. Please upload documents.")
