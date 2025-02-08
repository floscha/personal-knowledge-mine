# Check for inspiration on implementation:
from langchain_community.tools.tavily_search import TavilySearchResults

import lightcast
from langchain_core.tools import tool


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
