import lightcast
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
import lightcast.core

from tool_chain import build_tool_chain


@tool
def find_podcast(search_query: str) -> dict:
    """
    Returns the length of a word.

    Args:
        search_query (str):
    
    Returns:
        dict:
    
    """
    res = lightcast.search_podcasts(search_query)[0]
    return res.__dict__

from urllib.request import urlopen
from xml.dom import minidom
from lightcast.core import Episode


@tool
def list_podcast_episodes(feed_url: str) -> dict:
    "List all episode from a podcast given its feed url."
    xml_str = urlopen(feed_url).read()
    xmldoc = minidom.parseString(xml_str)
    episode_items = xmldoc.getElementsByTagName("item")
    episode_objects = [Episode.from_xml(item) for item in episode_items]

    return [e.__dict__ for e in episode_objects]


tools = [find_podcast, list_podcast_episodes]
model = ChatOllama(model="llama3.2")

chain = build_tool_chain(tools, model)
print(chain.invoke({"input":"Search for the huberman lab podcast"}))
print([e["title"] for e in chain.invoke({"input":"List all episodes for podcast https://feeds.megaphone.fm/hubermanlab"})["output"]])
