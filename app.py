import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# from langchain.llms import HuggingFaceHub
import os

st.set_page_config(page_title="Ask your PDF", page_icon="📄", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1614850523060-8da1d56ae167?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: #000000;  /* default text color black */
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }

    /* User message style */
    .user-msg {
        background-color: #cde4ff;
        color: #1e293b;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #94a3b8;
        max-width: 70%;
        word-wrap: break-word;
    }

    /* Bot message style */
    .bot-msg {
        background-color: #e9f2ff;
        color: #0f172a;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border: 1px solid #a5b4fc;
        max-width: 70%;
        word-wrap: break-word;
    }

    /* Download button styling */
    .stDownloadButton > button {
        color: white !important;
        background-color: #0f172a !important;
        border: none !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    
    [data-testid="stHeader"] * {
        color: white !important;
        fill: white !important;
    }

    header {
        box-shadow: none !important;
    }
    .stTextInput label,
    .stTextInput input::placeholder {
        color: black !important;
    }
    .stTextInput > div > div > input {
        color: black !important;
        background-color: #ffffff !important;
    }

    .stFileUploader label {
        color: black !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Ask your PDF")
# uploading the PDF
pdf=st.file_uploader("Upload your PDF",type="pdf")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_api_key"

#extracting the text of the pdf by looping through the pages
if pdf is not None:
    pdf_reader=PdfReader(pdf)
    text=""
    for page in pdf_reader.pages:
        text += page.extract_text()
    st.download_button("📥 Download Extracted Text", text, file_name="extracted_text.txt")
    #breaking text information in the pdf into chunks
    text_splitter=CharacterTextSplitter(separator="\n",chunk_size=1000,
                                        chunk_overlap=200,length_function=len)
    chunks=text_splitter.split_text(text)
    # st.write(chunks)

    #getting embeddings of the chunks
    embedding_model= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # embeddings = embedding_model.embed_documents(chunks)
    #creating vectorstore
    knowledge_base=FAISS.from_texts(chunks,embedding_model)

    #taking user query
    st.subheader("💬 Chat with your PDF")
    user_qn=st.text_input("Ask a qn from ur pdf")
    if user_qn:
        st.markdown(f"<div class='user-msg'><strong>You:</strong> {user_qn}</div>", unsafe_allow_html=True)
        docs=knowledge_base.similarity_search(user_qn,k=5)
        context = "\n".join([doc.page_content for doc in docs])

        # Use DistilBERT Q&A
        qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
        result = qa_pipeline(question=user_qn, context=context)
        response = result["answer"]
        st.write(response)
        st.markdown(f"<div class='bot-msg'><strong>PDF Assistant:</strong> {response}</div>",
                    unsafe_allow_html=True)
        st.download_button("📥 Download Answer", response, file_name="pdf_answer.txt")

        # # Wrap it in LangChain
        # llm = HuggingFacePipeline(pipeline=qa_pipeline)
        #
        # prompt_template = """
        # You are a helpful assistant. Use the following extracted parts of a document to answer the question.
        #
        # Only use the information from the provided context. If the answer is not in the context, say "I don't know".
        #
        # Context:
        # {context}
        #
        # Question:
        # {question}
        #
        # Helpful answer:
        # """
        #
        # # Define PromptTemplate
        # prompt = PromptTemplate(
        #     template=prompt_template,
        #     input_variables=["context", "question"]
        # )
        #
        # # Load QA chain with custom prompt
        # chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        #
        # # Run the chain
        # response = chain.run(input_documents=docs, question=user_qn)
        # st.write(response)
        # st.markdown(f"<div class='bot-msg'><strong>PDF Assistant:</strong> {response}</div>",
        #             unsafe_allow_html=True)
        #
        # # Download answer
        # st.download_button("📥 Download Answer", response, file_name="pdf_answer.txt")
