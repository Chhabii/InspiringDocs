import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain


from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss

from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


from dotenv import load_dotenv
import numpy as np
import pickle
import os


#sidebar contents
with st.sidebar:

    st.title("InpiringDocs")
    st.markdown(
        """
        This app is an llm-powered chatbot built using: 
        - [Streamlit](https://streamlit.io/)
        - [OpenAI](https://openai.com/)
        - [Langchain](https://langchain.com/)

        """
    )
    add_vertical_space(2)
    st.write("Made by Chhabi Acharya.")


# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []


def main():

    st.header("Chat with PDFs")
    load_dotenv()
    #upload a pdf
    pdf = st.file_uploader("Upload a PDF", type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        if not text:
            st.write("No text extracted from PDF.")
            return  # Early return if text extraction fails
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100, #overlaps 100 characters
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        st.write(f"Total chunks created: {len(chunks)}")  # Debug output for chunks

        # st.write(chunks)

        if not chunks:
            st.write("No chunks to process.")
            return

        # to reduce charges, we can save the vectorstores in a pickle file
        store_name = pdf.name[:-4]
        index_folder = f'src/faiss_store/{store_name}'

        if os.path.exists(index_folder):
            try:
                embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
                vectorstores = FAISS.load_local(index_folder, embeddings,allow_dangerous_deserialization=True)
                st.write("Loaded vectorstores from local storage")
            except Exception as e:
                st.write(f"Failed to load local storage: {e}")
        else:
            try: 
                # os.makedirs(index_folder)
                embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)

                faiss_index = faiss.IndexFlatL2(embeddings.dimensions)
                # Initialize the docstore
                docstore = InMemoryDocstore()
                # Initialize the index_to_docstore_id
                index_to_docstore_id = {}



                # vectorstores = FAISS(chunks,embeddings)
                vectorstores = FAISS(embedding_function=embeddings, index=faiss_index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

                # Add chunks to the vectorstores
                chunk_embeddings = embeddings.embed_documents(chunks)
                vectorstores.add_embeddings(list(zip(chunks, chunk_embeddings)))


                vectorstores.save_local(index_folder)
                st.write("Saved vectorstores to local storage")
            except Exception as e:
                st.write(f"Failed to save local storage: {e}")

        
        query = st.text_input("Ask a question about your PDF file: ")

        if query: 
            try:
                # docs = vectorstores.similarity_search(query,k=3)
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                compressor = LLMChainExtractor.from_llm(llm)
                base_retriever = vectorstores.as_retriever()

                compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
                
                docs = compression_retriever.get_relevant_documents(query)
                
                # st.write("Docs: ",docs)#debugging
                if docs:
                    #setup the memory and the chat prompt template
                    memory = ConversationBufferMemory(return_messages=True)
                    prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", "You are an intelligent assistant that helps answer questions based on the provided context: \n\n{context}\n\n Additionally, suggest related questions the user might ask. If you don't know the answer, kindly prompt the user to ask a more specific question about the document."),
                            ("human", "{question}")
                        ]
                    )
                     # Combine prompt and LLM using the new runnable pattern
                    chain = prompt | llm

                    memory_variables = memory.load_memory_variables({})
                    # st.write("Memory Variables: ",memory_variables) #debugging

                    #extract the context from the documents
                    context = "\n".join([doc.page_content for doc in docs])
                    # st.write("Context: ",context) #debugging

                    # Use the chain to invoke the response
                    response = chain.invoke({
                        'context': docs, 
                        'question': query,
                        **memory_variables
                    })
                    
                    # Save conversation to memory
                    memory.save_context({"question": query}, {"answer": response.content})

                    # st.write("Response text: ",response.content)
                    # st.write("Conversation history:", memory.buffer)
                    # memory_variables = memory.load_memory_variables({})
                    # st.write("Memory Variables: ",memory_variables) #debugging

                    # Save the interaction to session state
                    st.session_state.chat_history.append((query, response.content))


                else:
                    st.write("No documents found.")
            except Exception as e:
                st.write(f"Failed to search: {e}")
    # Display chat history
    st.write("### Chat History")
    for query, response in st.session_state.chat_history:
        st.markdown(f"**You:** {query}")
        st.markdown(f"**Bot:** {response}")


if __name__ == "__main__":
    main()