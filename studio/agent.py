"""
Example agent that implements LangGraph BigTool with Studio integration.
This example implements the agent from the README Quickstart but adds
support for LangGraph Studio's Store for cross-thread memory persistence.
"""

import math
import types
import uuid

from langchain.chat_models import init_chat_model

from langgraph_bigtool import create_agent
from langgraph_bigtool.utils import convert_positional_only_function_to_tool

from langgraph.store.base import BaseStore

# llm = init_chat_model("openai:gpt-4o-mini")
llm = init_chat_model("ollama:qwen2.5:14b")

def index_tools_in_store(state, store: BaseStore):

    all_tools = []
    for function_name in dir(math):
        function = getattr(math, function_name)
        if not isinstance(
            function, types.BuiltinFunctionType
        ):
            continue
        # This is an idiosyncrasy of the `math` library
        if tool := convert_positional_only_function_to_tool(
            function
        ):
            all_tools.append(tool)

    ## Create registry of tools. This is a dict mapping identifiers to tool instances.
    tool_registry = {
        str(uuid.uuid4()): tool
        for tool in all_tools
    }

    for tool_id, tool in tool_registry.items():
        store.put(
            ("tools",),
            tool_id,
            {
                "description": f"{tool.name}: {tool.description}",
            },
        )

builder = create_agent(llm, tool_registry)
agent_subgraph = builder.compile()