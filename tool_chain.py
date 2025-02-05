from typing import Any, Optional, TypedDict

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnablePassthrough
from langchain_core.tools import BaseTool, render_text_description


class ToolCallRequest(TypedDict):
    """A typed dict that shows the inputs into the invoke_tool function."""

    name: str
    arguments: dict[str, Any]
    

def build_tool_chain(tools: list[BaseTool], model):
    rendered_tools = render_text_description(tools)
    system_prompt = f"""
        You are an assistant that has access to the following set of tools. 
        Here are the names and descriptions for each tool:

        {rendered_tools}

        Given the user input, return the name and input of the tool to use. 
        Return your response as a JSON blob with 'name' and 'arguments' keys.

        The `arguments` should be a dictionary, with keys corresponding 
        to the argument names and the values corresponding to the requested values.
    """

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{input}")]
    )

    def invoke_tool(
        tool_call_request: ToolCallRequest, config: Optional[RunnableConfig] = None
    ):
        """A function that we can use the perform a tool invocation.

        Args:
            tool_call_request: a dict that contains the keys name and arguments.
                The name must match the name of a tool that exists.
                The arguments are the arguments to that tool.
            config: This is configuration information that LangChain uses that contains
                things like callbacks, metadata, etc. See LCEL documentation about RunnableConfig.

        Returns:
            output from the requested tool
        """
        tool_name_to_tool = {tool.name: tool for tool in tools}
        name = tool_call_request["name"]
        requested_tool = tool_name_to_tool[name]
        return requested_tool.invoke(tool_call_request["arguments"], config=config)

    return prompt | model | JsonOutputParser() | RunnablePassthrough.assign(output=invoke_tool)
