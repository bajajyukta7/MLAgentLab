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
            Take an input image from the user, analyze it using a function for classification, and return the output as one of the specified categories: "cat," "dog," or "none."

            Provide a detailed response based on whether the image matches a classification.

            # Steps

            1. Accept an input image from the user.
            2. Pass the image to a pre-defined classification function (e.g., `model_test`).
            3. Evaluate the classification function's result.
            - If the result matches "cat," return "Classified as: cat."  
            - If the result matches "dog," return "Classified as: dog."  
            - If neither applies, return "Classified as: none."
            4. Ensure the classification is accurate and succinct in the response.

            # Output Format

            The output should be a single string specifying the classification. For example:
            - `"Classified as: cat"`
            - `"Classified as: dog"`
            - `"Classified as: none"`

            # Notes

            - Ensure the model's response is consistent with the output from the classification function.
            - Provide feedback if no image is supplied or if the input is invalid.
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