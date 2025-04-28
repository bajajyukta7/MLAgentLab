from tool_wrapper import ToolWrapper
import os  
import dotenv
import json
import re
# import user_proxy_agent

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_core import RoutedAgent
from autogen_agentchat.messages import HandoffMessage

# Global variables
retriever = None
agent = None
system_message = None

from typing import AsyncGenerator, Sequence
import json
import re
from autogen_core import CancellationToken
from autogen_agentchat.messages import BaseMessage
from autogen_agentchat.messages import (
    AgentEvent,
    BaseChatMessage,
    ChatMessage,
    ModelClientStreamingChunkEvent,
    TextMessage
)

from autogen_agentchat.base import Response


class DataProcessingAgent():
    def get_agent(model_client, tools=[]):
        system_prompt = """
            You are a premium AI assistant, your job is to process the data and resize the images.
            Call function resize images resize_images.
        """
        
        # print("tools: ", tools)
        # Initialize Assistant Agent
        mle_agent = AssistantAgent(
            name="MLEAgent",
            description="An agent for training the ML model and evaluating it.",
            tools=tools,
            model_client=model_client,
            system_message=system_prompt,
            handoffs=["user"],
            reflect_on_tool_use = True
        )
        
        return mle_agent