from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings



# Step 1: Extract data from the PDF
def load_pdf_file(data_path):
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()
    return documents

# Step 2: Split the text into smaller chunks
def text_split(extracted_data):
    # We split text into chunks of 500 characters so the AI can process small pieces at a time
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Step 3: Download the embedding model from Hugging Face
def download_hugging_face_embeddings():
    # This model converts text into numerical vectors that Pinecone understands
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings
