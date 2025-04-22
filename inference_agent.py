from tool_wrapper import ToolWrapper
import os  
import json
import re

from autogen_agentchat.agents import AssistantAgent
# Global variables
retriever = None
agent = None
system_message = None

from typing import AsyncGenerator, Sequence
import json
import re

class InferenceAgent():
    def get_agent(model_client, tools=[]):
        system_prompt = """
            Take an input image from the user, analyze it using the model_test(image) function, and return the output as one of the categories: "cat", "dog", or "none".

            Steps:
            Accept an input image from the user (as a PIL.Image object).

            Pass it to the model_test function.

            Interpret the result:

            If the output is "Classified as: cat", return it as-is.

            If the output is "Classified as: dog", return it as-is.

            If the output doesn't match either, or there’s an error, return "Classified as: none".

            Output:
            Always return exactly one of the following strings:

            "Classified as: cat"

            "Classified as: dog"

            "Classified as: none"

            Notes:
            If no valid image is supplied, return "Classified as: none".

            Do not attempt to explain or elaborate in the output — just return the classification result string.
        """
        
        # print("tools: ", tools)
        # Initialize Assistant Agent
        mle_agent = AssistantAgent(
            name="InferenceAgent",
            description="An agent for training the ML model and evaluating it.",
            tools=tools,
            model_client=model_client,
            system_message=system_prompt,
            handoffs=["user"],
            reflect_on_tool_use = True
        )
        
        return mle_agent