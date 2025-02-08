from langchain_ollama import ChatOllama

from tool_chain import build_tool_chain
from lightcast_tools import find_podcast, list_podcast_episodes


tools = [find_podcast, list_podcast_episodes]
model = ChatOllama(model="llama3.2")

chain = build_tool_chain(tools, model)
print(chain.invoke({"input":"Search for the huberman lab podcast"}))
all_episodes = chain.invoke({"input":"List all episodes for podcast https://feeds.megaphone.fm/hubermanlab"})["output"]
print([e["title"] for e in all_episodes[:10]])
