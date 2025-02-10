from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders.generic import GenericLoader
from langchain_mlx_whisper_parser import MlxWhisperParser
from langchain_ollama import ChatOllama, OllamaLLM

from lightcast_tools import find_podcast, list_podcast_episodes
from podcast_audio_loader import PodcastAudioLoader
from tool_chain import build_tool_chain


tools = [find_podcast, list_podcast_episodes]
model = ChatOllama(model="llama3.2")

chain = build_tool_chain(tools, model)
# TODO: Check why this translates to {'search_query': 'Jagtenå p¥ det eviga liv'}
# foo = chain.invoke({"input": "Search for the Jagten på det evige liv podcast"})

podcast = chain.invoke({"input": "Search for the Jagten pa det evige liv podcast"})["output"]
feed_url = podcast["feed_url"]
all_episodes = chain.invoke({"input": f"List all episodes for podcast {feed_url}"})["output"]

audio_urls = [e["audio_url"] for e in all_episodes][:1]

loader = GenericLoader(PodcastAudioLoader(audio_urls, f"./{podcast['name']}"), MlxWhisperParser(cache_transcriptions=True))
docs = loader.load()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a concise English summary of the following:\\n\\n{context}")
])
chain = create_stuff_documents_chain(OllamaLLM(model="llama3.2"), prompt)

# Invoke chain
result = chain.invoke({"context": docs})
print(result)