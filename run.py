#!/usr/bin/env -S uv run
#
# /// script
# requires-python = "==3.10"
# dependencies = [
#   "langchain",
#   "langchain_community",
#   "langchain-ollama",
#   "transformers",
#   "torch",
#   "faiss-cpu",
#   "yt_dlp",
#   "pydub",
#   "librosa",
# ]
# ///

from langchain.chains import RetrievalQA
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParserLocal
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Two Karpathy lecture videos
urls = ["https://youtu.be/kCc8FmEb1nY", "https://youtu.be/VMj-3S1tku0"]

# Directory to save audio files
save_dir = "~/Downloads/YouTube"

# Transcribe the videos to text
loader = GenericLoader(YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParserLocal())
docs = loader.load()

# Combine doc
combined_docs = [doc.page_content for doc in docs]
text = " ".join(combined_docs)

# Split them
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
splits = text_splitter.split_text(text)

# Build an index
embeddings = OllamaEmbeddings(model="llama3.2")
vectordb = FAISS.from_texts(splits, embeddings)
# Build a QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OllamaLLM(model="llama3.2"),
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
)

# Ask a question!
qa_chain.run("Why do we need to zero out the gradient before backprop at each step?")
