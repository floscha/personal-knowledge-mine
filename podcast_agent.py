import lightcast
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from tool_chain import build_tool_chain


@tool
def find_podcast(search_query: str) -> dict:
    "Returns the length of a word."
    res = lightcast.search_podcasts(search_query)[0]
    return res.__dict__


@tool
def list_podcast_episodes(feed_url: str) -> dict:
    "List all episode from a podcast given its feed url."
    episode_objects = lightcast.core.get_episodes_from_feed_url(feed_url)

    return [e.__dict__ for e in episode_objects]


tools = [find_podcast, list_podcast_episodes]
model = ChatOllama(model="llama3.2")

chain = build_tool_chain(tools, model)
print(chain.invoke({"input":"Search for the huberman lab podcast"}))
all_episodes = chain.invoke({"input":"List all episodes for podcast https://feeds.megaphone.fm/hubermanlab"})["output"]
print([e["title"] for e in all_episodes[:10]])
