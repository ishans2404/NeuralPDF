import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDFs
def extract_pdf_text(pdfs):
    all_text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            all_text += page.extract_text()
    return all_text

# Function to split text into chunks
def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=12000, chunk_overlap=1200)
    text_chunks = splitter.split_text(text)
    return text_chunks

# Function to create vector store
def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to setup conversation chain for QA
def setup_conversation_chain(template):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input based on selected mode
def handle_user_input(mode, user_question=None):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    indexed_data = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = indexed_data.similarity_search(user_question)

    chain = setup_conversation_chain(prompt_template[mode])
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]      

# Prompt templates for each mode
prompt_template = {
"chat":"""
Your alias is Neural-PDF. Your task is to provide a thorough response based on the given context, ensuring all relevant details are included. 
If the requested information isn't available, simply state, "answer not available in context," then answer based on your understanding, connecting with the context. 
Don't provide incorrect information.\n\n
Context: \n {context}?\n
Question: \n {question}\n

Answer:
""", 
"quiz":"""
Your alias is Neural-PDF. Your task is to generate multiple choice questions for quiz based on the given context and requested number of questions, ensuring all relevant details are included. 
If the requested information isn't available, simply state, "answer not available in context," then answer based on your understanding, connecting with the context. 
Don't provide incorrect information.\n\n
Context: \n {context}?\n
Question: \n {question}\n

Answer:
""", 
"long":"""
Your alias is Neural-PDF. Your task is to generate long answer-type questions based on the given context and requested number of questions, ensuring all relevant details are included. 
If the requested information isn't available, simply state, "answer not available in context," then answer based on your understanding, connecting with the context. 
Don't provide incorrect information.\n\n
Context: \n {context}?\n
Question: \n {question}\n

Answer:
""", 
}


# Streamlit app
def main():
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "mode" not in st.session_state:
        st.session_state.mode=""
    if "file_upload" not in st.session_state:
        st.session_state.file_upload=False
    
    st.set_page_config(page_title="NeuralPDF", page_icon=":page_with_curl:", initial_sidebar_state="expanded", layout="wide")
    st.title("NeuralPDF: Interactive PDF Chat using AI 🤖")
    
    # sidebar
    files = st.sidebar.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)
    if st.sidebar.button("Submit"):
        if files:
            with st.spinner("Processing..."):
                raw_text = extract_pdf_text(files)
                text_chunks = split_text_into_chunks(raw_text)
                create_vector_store(text_chunks)
            st.sidebar.success("Processing done!")
            st.session_state.file_upload=True

    # mode of chat
    with st.sidebar:
        if st.session_state.file_upload:
            # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
            # st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)
            modes={"Chat Conversation":"chat", "Quiz & MCQs":"quiz", "Long-Answer Questions":"long"}
            choose_mode = st.radio("", list(modes.keys()), index=0)
            st.session_state.mode=modes[choose_mode]

    if st.session_state.file_upload:    
        # keep history of chat
        for dialogue in st.session_state.conversation:
            with st.chat_message(dialogue["role"]):
                if st.session_state.mode != "chat" and dialogue["role"] == "assistant":
                    st.markdown(dialogue["content"])
                    with st.expander("Answer"):
                        st.markdown(dialogue["answer"])
                else: st.markdown(dialogue["content"])

        # handle conversation
        if prompt := st.chat_input("Type your question here"):
            # handle user side
            with st.chat_message("user"): st.markdown(prompt)
            st.session_state.conversation.append({"role":"user", "content":prompt, "answer":""})
            # handle assistant side
            with st.chat_message("assistant"):
                response=handle_user_input(st.session_state.mode, prompt)
                answer=""
                if st.session_state.mode != "chat":
                    answer = handle_user_input("chat", response)
                    st.markdown(response)
                    with st.expander("Answer"):
                        st.markdown(answer)
                else: st.markdown(response)
            st.session_state.conversation.append({"role":"assistant", "content":response, "answer":answer})


# Launch the app
if __name__ == "__main__":
    main()