# Check for inspiration on implementation:
from langchain_community.tools.tavily_search import TavilySearchResults

import lightcast
from langchain_core.tools import tool

# class PodcastNotFoundException

@tool
def find_podcast(search_query: str) -> dict:
    "Returns the length of a word."
    search_results = lightcast.search_podcasts(search_query)
    if len(search_results) > 0:
        return search_results[0].__dict__
    return {}


@tool
def list_podcast_episodes(feed_url: str) -> dict:
    "List all episode from a podcast given its feed url."
    episode_objects = lightcast.core.get_episodes_from_feed_url(feed_url)

    return [e.__dict__ for e in episode_objects]
