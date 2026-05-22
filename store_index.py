from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

# 1. Load the secret API key from our .env file
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

if not PINECONE_API_KEY or PINECONE_API_KEY == 'your-actual-api-key-goes-here':
    raise ValueError("Don't forget to put your actual Pinecone API key in the .env file!")

# 2. Extract and Process the Data using the helper functions we built
print("Extracting text from PDF...")
extracted_data = load_pdf_file(data_path='Data/')

print("Splitting text into chunks...")
text_chunks = text_split(extracted_data)

print("Downloading Embedding Model...")
embeddings = download_hugging_face_embeddings()

# 3. Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot" # We can name our database anything

# 4. Create the Index (Database Table) if it doesn't exist
existing_indexes = pc.list_indexes().names()
if index_name not in existing_indexes:
    print(f"Creating new Pinecone index: '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384, # The miniLM model creates vectors of size 384
        metric="cosine", # We use cosine similarity to find the closest matches
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print(f"Index '{index_name}' already exists. Skipping creation.")

# 5. Push all our processed chunks into the Pinecone Database!
print("Uploading vectors to Pinecone. This might take a few minutes...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print(f"Success! Uploaded {len(text_chunks)} text chunks to Pinecone.")
