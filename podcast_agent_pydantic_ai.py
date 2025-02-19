import lightcast

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel


ollama_model = OpenAIModel(
    model_name="llama3.2", base_url="http://localhost:11434/v1", api_key="FAKE_KEY"
)
agent = Agent(
    model=ollama_model,
    system_prompt=(
        "Use the `search_podcast` tool retrieve a list of podcasts together with their feed url from which a full list of episodes can be retrieved."
        "Then, use the `get_episodes` tool to get the full list of episodes for the most relevant podcast from the list above."
        "Finally, print the title of the 5 most relevant episodes."
    ),
)


@agent.tool
async def search_podcast(_: RunContext, query: str) -> str:
    """Search podcasts based on the given query.

    Args:
        ctx: The context.
        query: A textual query to search podcasts based upon.
    """
    return "\n".join(
        f"{e.name} ({e.feed_url})" for e in lightcast.search_podcasts(query)[:10]
    )


@agent.tool
async def get_episodes(_: RunContext, feed_url: str) -> str:
    """Search podcasts based on the given query.

    Args:
        ctx: The context.
        query: The feed url of the podcast for which to get all episodes.
    """

    return "\n".join(
        f"{e.title} ({e.audio_url})"
        for e in lightcast.core.get_episodes_from_feed_url(feed_url)
    )


result = agent.run_sync("Find all episodes of the Huberman Lab podcast")
print(result.data)
# print("\n", result.usage())
