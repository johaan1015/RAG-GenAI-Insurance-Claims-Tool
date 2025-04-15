import streamlit as st
import PyPDF2
import json
from langchain_community.chat_models import ChatOpenAI
import os
from streamlit_option_menu import option_menu
from functionalities import liability_des, eligibility_check, summary_of_docs,q_a
import ast


os.environ["OPENAI_API_KEY"] = "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"
def main():
    st.title("Insurance Claims Dashboard")

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



    uploaded_files = st.file_uploader("Upload PDF documents. Please upload Claim Form first", type="pdf", accept_multiple_files=True)

    if uploaded_files:
            st.session_state['uploaded_files'] = uploaded_files
    with st.sidebar:
        image_path="Perceptive_Analytics_logo.jpg"
        st.image(image_path, use_column_width=True)
        selected = option_menu("Main Menu",["Liability Decision", "Eligibility Check", "Summary","Q&A"],
                            icons=['file-earmark', 'check-circle', 'list','question-circle'],
                            menu_icon="cast", default_index=0)
        
        


    if selected == "Liability Decision":
        liability_des.show()
    elif selected == "Eligibility Check":
        eligibility_check.show()
    elif selected == "Summary":
        summary_of_docs.show()
    elif selected == "Q&A":
         q_a.show()


if __name__=='__main__':
     main()
