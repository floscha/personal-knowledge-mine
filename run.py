from langchain.chains import RetrievalQA
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mlx_whisper_parser import MlxWhisperParser


# List of YouTube URLs to download audio from
urls = ["https://youtu.be/TVUibwoVXZc"]

# Directory to save audio files
save_dir = "~/Downloads/YouTube"

# Transcribe the videos to text
loader = GenericLoader(YoutubeAudioLoader(urls, save_dir), MlxWhisperParser())
docs = loader.load()

# Combine doc
combined_docs = [doc.page_content for doc in docs]
text = " ".join(combined_docs)

# Split them
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
splits = text_splitter.split_text(text)

# Build an index
embeddings = OllamaEmbeddings(model="llama3.2")
vectordb = InMemoryVectorStore.from_texts(splits, embeddings)
# Build a QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OllamaLLM(model="llama3.2"),
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
)

# Ask a question!
result = qa_chain.invoke("How should I handle my mobile phone with regards to dopamine?")["result"]
print(result)
