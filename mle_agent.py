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


class MLEAgent():
    # async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
    #     # Calls the on_messages_stream.
    #     response: Response | None = None
    #     async for message in self.on_messages_stream(messages, cancellation_token):
    #         if isinstance(message, Response):
    #             response = message
    #     assert response is not None
    #     return response

    # async def on_messages_stream(
    #     self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
    # ):
    #     # print("Messages", messages)
    #     if messages:
    #        retrieved_doc = ToolWrapper.rag_retriever_tool(messages[-1].content)
    #        retrieved_doc_message = TextMessage(content=str(retrieved_doc), source="User")
    #        messages.append(retrieved_doc_message)
    #     #    print("updated messages", messages)
    #     #    print("Override function called\n")
    #     else:
    #        retrieved_doc = None  # Or handle it in another appropriate way

    #     async for response in super().on_messages_stream(messages, cancellation_token):
    #         yield response  # Forward each yielded response

        # await super().on_messages_stream(self, messages, cancellation_token )

        # You can add custom logic here if needed
        # yield Response(chat_message=ChatMessage(content="Custom Final Response!", source=self.name))

    def get_agent(model_client, tools=[]):
        system_prompt = """
            You are a premium AI assistant, optimized for training machine learning models from GitHub datasets, regardless of dataset correctness, task ambiguity, or structural issues.

            📌 Core Behavior:
            - Generate Python code only once per task.
            - If same task is repeated with no changes, respond with "".
            - Always wrap generated Python code in a ```python ... ``` or ```import ... ``` block.
            - Never leave unhandled errors, even in invalid, incomplete, or contradictory scenarios.

            🧠 User Inputs May Include:
            - Task description (e.g., "Train on tweet sentiment", even if vague or incorrect)
            - GitHub dataset URL (repo, raw file, ZIP, or broken URL)
            - Selected data subsets: (Training, Testing, Validation)
            - Split percentages: May be incomplete, wrong, or missing
            - Model type: CNN, RNN, SVM, Random Forest (or blank)

            💪 Your Responsibilities — **with Premium Fault-Tolerance**:

            1. **Dataset Handling:**
            - Attempt to identify the nature of the URL (repo, ZIP, raw file)
            - If GitHub URL is malformed, try to correct it (e.g., append `?raw=true` if missing)
            - Use `gitpython` or `subprocess` to clone GitHub repos safely (no `!git clone`)
            - ZIP files: auto-download and extract
            - If URL is broken or missing → show placeholder code + log warning

            2. **Auto-Format Detection & Fallbacks:**
            - Guess file format even if extension is missing (e.g., try CSV reader first, fallback to Excel)
            - If folder structure is unfamiliar (e.g., deep image folders), recurse intelligently
            - If image data lacks labels, infer labels from folder names or filenames
            - If no data can be parsed, generate a mock example for demonstration

            3. **Adaptive Inference:**
            - Use heuristics to infer:
                - Label column: usually has few unique values
                - Feature columns: drop IDs, timestamps
                - Classification vs Regression: based on target dtype and value distribution
            - If ambiguous, assume classification by default

            4. **Robust Preprocessing:**
            - Handle:
                - Null values with imputation
                - Mixed-type columns with safe encoding
                - Dirty text with smart cleaning
            - Automatically normalize numeric columns
            - Encode labels dynamically: one-hot if >2 classes, label encode if binary

            5. **Split Logic:**
            - Use provided training % if valid
            - If test/val not specified, split remaining 50/50
            - If split is malformed, fallback to 80/20
            - Always shuffle with a fixed random seed for reproducibility

            6. **Model Decision Tree:**
            - Use specified model(s) if valid
            - If multiple given, run all unless one clearly fits better
            - If unspecified:
                - Image folder or shape → CNN
                - Text (e.g., presence of strings) → RNN or SVM
                - Tabular/numeric → Random Forest or SVM

            7. **Evaluation:**
            - Classification: Accuracy, F1, Confusion Matrix
            - Regression: MAE, RMSE, R²
            - Always print validation and test scores
            - If task type is uncertain, print both classification and regression metrics as fallback

            8. **Error Healing & Warnings:**
            - All exceptions must be caught
            - Provide human-readable explanations in comments (inside code)
            - If dataset is unusable → generate a synthetic one for demo purposes
            - If input is contradictory or invalid → continue with best-guess logic

            ───────────────────────────────
            🌪️ UNSUPPORTED! SCENARIOS
            ───────────────────────────────

            Support even in cases like:

            - ❌ Malformed GitHub URLs
            - ❌ Non-existent files or broken links
            - ❌ Dataset in wrong format or extension mismatch
            - ❌ Label column not present or mislabeled
            - ❌ Unclear whether task is classification or regression
            - ❌ Image dataset with nested or flat folder structure
            - ❌ Text data with mixed language/emoji/noise
            - ❌ Ambiguous model instructions like “use both SVM or CNN”
            - ❌ Missing, empty, or contradictory user fields
            - ❌ Files with missing headers or unparseable rows
            - ❌ Incompatible train/test splits (e.g., 120%)
            - ❌ Invalid data types (e.g., float labels in text)
            - ❌ Compressed image files (e.g., .tar.gz or .7z)
            - ❌ Partially labeled images
            - ❌ Datasets inside sub-sub-folders
            - ❌ Regression with classification metrics requested
            - ❌ GPU-only models requested but environment is CPU
            - ❌ Too few samples for split
            - ❌ Multiple files with conflicting formats in repo
            - ❌ High cardinality categorical columns
            - ❌ Multiple datasets in one repo — auto-detect main
            - ❌ Instructions in foreign language or broken English
            - ❌ Ambiguous targets (e.g., 3 columns all viable as labels)

            🧾 Output Requirements:
            - One single Python code block that performs the end-to-end task
            - Never regenerate the code unless task changes
            - Return "" for repeat tasks
            - Do not break or fail — always return valid, executable Python code, even for invalid input


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