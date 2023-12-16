import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain

from dotenv import load_dotenv, find_dotenv
load_dotenv()


st.title("News Research Tool📈")
st.sidebar.title("News Article URLs")

urls=[]
for i in range(3):
    url=st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)


process_url_clicked=st.sidebar.button("Process URLs")

main_placefolder=st.empty()

llm=OpenAI(temperature=.7,max_tokens=500)
file_path='faiss_store_openai.pkl'
if process_url_clicked:
    # Loading data from urls
    loader=UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data loading...Started...✅✅✅")
    data=loader.load()

    # Splitting the data into chunks
    text_splitter=RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',','],
        chunk_size=500)
    main_placefolder.text("Data splitting...Started...✅✅✅")
    docs=text_splitter.split_documents(data)
    

    # Creating Embeddings and save it to FAISS index
    embeddings=OpenAIEmbeddings()
    vectorstore_openai=FAISS.from_documents(docs,embeddings)
    main_placefolder.text("Data Embedding...Started...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    
    with open(file_path, 'wb') as f:
        pickle.dump(vectorstore_openai,f)

query=main_placefolder.text_input("Question: ")
if query:
    
    if os.path.exists(file_path):
        with open(file_path,'rb') as f:
            vectorstore=pickle.load(f)
            chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vectorstore.as_retriever())
            result=chain({"question": query},return_only_outputs=True)
            #{"answer": "","sources": []}
            st.header("Answer")
            st.subheader(result['answer'])

            #Display source urls if available

            sources=result.get("sources","")
            if sources:
                st.subheader("Sources:")
                sources_list=sources.split("\n")  #Split the sources by newline
                for source in sources_list:
                    st.write(source)


    
