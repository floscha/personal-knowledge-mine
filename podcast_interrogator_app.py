import streamlit as st
from lightcast import search_podcasts
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mlx_whisper_parser import MlxWhisperParser

from podcast_audio_loader import PodcastAudioLoader


st.title("🎙️ Podcast Interrogator")

podcast = search_podcasts("huberman lab")[0]
first_episode = podcast.episodes[-1]
urls = [first_episode.audio_url]

loader = GenericLoader(PodcastAudioLoader(urls, "./"), MlxWhisperParser())
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


def generate_response(input_text: str):
    result = qa_chain.invoke(input_text)["result"]
    st.markdown(result)


with st.form("llm_form"):
    text = st.text_area("Enter text:",
        # "What are the three key pieces of advice for learning how to code?",
    )
    submitted = st.form_submit_button("Submit")
    if submitted: generate_response(text)
