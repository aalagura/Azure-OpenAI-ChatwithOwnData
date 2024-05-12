import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from PyPDF2 import PdfReader
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from prompt import SYSTEM_MESSAGE
from langchain.embeddings import AzureOpenAIEmbeddings

load_dotenv()

ENDPOINT_URL = os.environ.get("AZURE_OPENAI_ENDPOINT")
API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
API_VERSION = "2024-02-01"
OPENAI_EMBEDDING_ENGINE = "text-embedding-ada-002"
COMPLETION_MODEL = 'gpt-4'

client = AzureOpenAI(azure_endpoint=ENDPOINT_URL,api_key=API_KEY,api_version=API_VERSION)  
# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="Files/Chromadb")

def get_pdf_text(pdf_docs):
    text="" 
    
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf_docs)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return chunks

def get_embedding(text):
    return client.embeddings.create(input=text, model=OPENAI_EMBEDDING_ENGINE).data[0].embedding

def get_vector_store(text_chunks):   

    # Create a new Chroma collection
    collection_name = "my_collection"
    collection = chroma_client.get_or_create_collection(name=collection_name)

    docid=[]
    embeds = []
    for i, chunk in enumerate(text_chunks):
          embeds.append(get_embedding(chunk))
          docid.append(f"doc_{i}")

    # Embed the text using Azure OpenAI embeddings
    #documents = [{"page_content": text_chunks, "metadata": {"source": "Files/Description.pdf"}}]
    # Store the embeddings in Chroma
    collection.upsert( embeddings=embeds,ids=docid)
    return collection

def get_chat_completion(results,question):
    messages = [
        {'role': 'system', 'content': SYSTEM_MESSAGE},
        {'role': 'user', 'content': "knock knock"}
    ]

    prompts = []
    
    for r in results:
        # construct prompts based on the retrieved text chunks in results 
        prompt = f"{'role': 'user', 'content': 'Please extract the following: {question}  solely based on the text below. Use an unbiased and journalistic tone. If you're unsure of the answer, say you cannot find the answer. \n\n {r}}"
        prompts.append(prompt)    

    response = client.chat.completions.create(
        model="gpt-4",
        messages=prompts,
        temperature=0, 
        max_tokens=500
    )
    
    print(response.choices[0].message.content)

def main():
  #  st.set_page_config("Chat PDF")
   # st.header("Chat with PDF using azure open ai")

    #user_question = st.text_input("Ask a Question from the PDF Files")

    #if user_question:
     #   user_input(user_question)

    #with st.sidebar:
     #   st.title("Menu:")
     #   pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
     #   if st.button("Submit & Process"):
     #       with st.spinner("Processing..."):
                raw_text = get_pdf_text("Files/Description.pdf")
                
                text_chunks = get_text_chunks(raw_text)
                #print(text_chunks)
                collec = get_vector_store(text_chunks)

                #Create a new Chroma collection
                #collection_name = "my_collection"
                #collec = chroma_client.get_or_create_collection(name=collection_name)
                ques = "what is the job description?"
                q_emd = get_embedding(ques)
                q= collec.query(query_embeddings=q_emd,n_results=2)
                results = q["documents"][0]
                get_chat_completion(results,ques)
                #user_input("who are you")
     #           st.success("Done")



if __name__ == "__main__":
    main()