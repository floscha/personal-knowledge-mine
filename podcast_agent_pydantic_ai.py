import lightcast

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel


ollama_model = OpenAIModel(
    model_name="llama3.2", base_url="http://localhost:11434/v1", api_key="FAKE_KEY"
)
agent = Agent(
    model=ollama_model,
    system_prompt=(
        "Reply in one concise sentence."
        "Use the `search_podcast` tool to find the most relevant result."),
)

@agent.tool
async def search_podcast(_: RunContext, query: str) -> str:
    """Search podcasts based on the given query.

    Args:
        ctx: The context.
        query: A textual query to search podcasts based upon.
    """
    return lightcast.search_podcasts(query)[0].name


result = agent.run_sync("Huberman Lab")
print(result.data)
print("\n", result.usage())
