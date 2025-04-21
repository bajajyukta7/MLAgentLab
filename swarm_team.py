from agent_wrapper import AgentWrapper
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from agent_wrapper import AgentWrapper

import os
from autogen_agentchat.teams import Swarm
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from typing import Sequence
from autogen_agentchat.messages import AgentEvent, ChatMessage
import re
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.messages import ToolCallExecutionEvent


def custom_speaker_selection(last_speaker, groupchat, config_validation_agent,observability_agent,workloads_agent):
    # print("mesvbvsage:", messages)
    last_message = ""
    # last_message = messages[-1].content.lower()
    # last_user_message = re.findall(r"User:\s*(.+)", last_message, re.DOTALL)

    # # Return the last user message if found, otherwise return an empty string
    # # last_message = last_user_message[-1].strip() if messages else ""
    # print("last message:", last_user_message)
    # Define keywords associated with each agent's expertise
    config_validation_keywords = ['quality check', 'config validation']
    observability_keywords = ['observability', 'monitoring', 'health']
    workload_keywords = ['design', 'workload definition', "virtual instance", "vi"]

    # Determine the appropriate agent based on the presence of keywords
    if any(keyword in last_message for keyword in workload_keywords):
        return "WorkloadsAssistant"
    elif any(keyword in last_message for keyword in config_validation_keywords):
        return "ConfigValidationAssistant"
    elif any(keyword in last_message for keyword in observability_keywords):
        return "ObservabilityAssistant"
    else:
        return "WorkloadsAssistant"

def selector_func_with_user_proxy(messages: Sequence[AgentEvent | ChatMessage]) -> str | None:
   
    print("mesvbvsage:", messages)
    last_message = ""
    # last_message = messages[-1].content.lower()
    # last_user_message = re.findall(r"User:\s*(.+)", last_message, re.DOTALL)

    # # Return the last user message if found, otherwise return an empty string
    # # last_message = last_user_message[-1].strip() if messages else ""
    # print("last message:", last_user_message)
    # Define keywords associated with each agent's expertise
    config_validation_keywords = ['quality check', 'config validation']
    observability_keywords = ['observability', 'monitoring', 'health']
    workload_keywords = ['design', 'workload definition', "virtual instance", "vi"]

    # Determine the appropriate agent based on the presence of keywords
    if any(keyword in last_message for keyword in workload_keywords):
        return "WorkloadsAssistant"
    elif any(keyword in last_message for keyword in config_validation_keywords):
        return "ConfigValidationAssistant"
    elif any(keyword in last_message for keyword in observability_keywords):
        return "ObservabilityAssistant"
    else:
        return "WorkloadsAssistant"

def initialize():

    # user_proxy = user_proxy_agent.TrackableUserProxyAgent(
    #     name="UserAgent",
    #     description="You are the human user. You must provide input for each interaction.",
    #     # human_input_mode="ALWAYS"
    # )
    
    # code_executor = CodeExecutorAgent(
    # name="code_executor",
    # code_executor=LocalCommandLineCodeExecutor(virtual_env_context=venv_context, work_dir=work_dir),
    # )
    # Get AI agents
    agents = [agent for agent in AgentWrapper.get_agents()]
    # agents.insert(0, user_proxy) 
  
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=2, include_agent_event=True)
    termination = text_mention_termination | max_messages_termination
    termination1 = HandoffTermination(target="user") | TextMentionTermination("TERMINATE") | (MaxMessageTermination(max_messages=6))

    selector_prompt = """
    Select an agent to perform task.

    {roles}

    Current conversation context:
    {history}

    Read the above conversation, then select an agent from {participants} to perform the next task.
    After an agent responds, return the response, and wait for input from user.
    Only select one agent and stop the conversation strictly and wait for user input.
    """
   
    selector_prompt_1 = """
    Always call the selector_func.
    Based on the last user input after patsing the string user:, select the most appropriate agent to respond next.

    Available agents:

    WorkloadsAssistant:
    - Handles general inquiries and topics related to the design of Workloads, Virtual Instances, and Workload Definitions.

    ConfigValidationAssistant:
    - Addresses questions related to quality checks or configuration validation.

    ObservabilityAgent:
    - Responds to queries concerning observability, monitoring, and workload health.

    Output only the selected agent's name.
    """
    selector_prompt_3= """
    Given a conversation history string, extract the last message and determine the next speaker.

    :param conversation_history: A multi-line string containing the chat history.
    :return: A tuple (last_message, next_speaker)

    Example:
    message: [TextMessage(source='user', models_usage=None, metadata={}, content='User: hi\nAssistant: WorkloadsAssistant:How can I assist you in designing your Azure workload today?\nUser: design a workload)]
    reponse: speaker selection based on "design a workload"
    """
    selector_prompt_with_react = """
    You run in a loop of Thought, Action, PAUSE, Observation.
    At the end of the loop you output an Answer
    Use Thought to describe your thoughts about the question you have been asked.
    Use Action to run one of the agents {roles} available to you - then return PAUSE.
    Observation will be the result of running those agents.
    Do not select random agents strictly.
    Think before responding

    Your available agents are:

    WorkloadsAssistant:
    This handle all generic questions and is the main agent.
    It will also handle question related to Virtual Instand and Workload Definition.
    
    ConfigValidationAssistant:
    Answer question related quality checks.

    ObservabilityAgent:
    Answer question related to observability and monitoring.

    Example session:

    Question: Design a workload?
    Thought: I should look for agent for designing and creating a workloads
    Action: WorkloadsAssistant
    PAUSE

    You will be called again with this:

    Observation: Return the response of workload assistant.

    You then output:

    Answer: WorkloadAssistant: "response of workload assistant"
    """

    selector_group_chat = SelectorGroupChat(
        agents,
        model_client=AgentWrapper.get_model_client(),
        selector_func=selector_func_with_user_proxy,
        termination_condition=termination1,
        selector_prompt=selector_prompt_3,
        max_selector_attempts = 1,
        allow_repeated_speaker=False,  # Allow an agent to speak multiple turns in a row.
    )
    # termination1 = HandoffTermination(target="user") | TextMentionTermination("TERMINATE") | MaxMessageTermination(max_messages=1)
    swarm_teams_manager = Swarm(
        participants = agents,
        # model_client=AgentWrapper.get_model_client(),
        termination_condition=termination1,
        max_turns=10
        # selector_prompt=selector_prompt_1,
        # allow_repeated_speaker=False,  # Allow an agent to speak multiple turns in a row.
    )
    
    return swarm_teams_manager

async def chat_with_agent(user_input):

    # if not input_prompt or not isinstance(input_prompt, str):
    #     raise ValueError("❌ Invalid input: Message must be a non-empty string.")
    messages = []
    # Convert conversation history into formatted messages
    for message in user_input:
        role = message.get("role", "").capitalize()  # Ensure proper casing (User/Assistant)
        content = message.get("content", "")
        messages.append(f"{role}: {content}")

        # Extract latest user message for context retrieval
        # latest_user_message = user_input[-1].get("content", "")
    latest_user_message = user_input[-1].get("content", "")
    print("Starting chat with input:", messages)
    messages_string = "\n".join(messages)

    agents = [agent for agent in AgentWrapper.get_agents()]

    
    selector_group_chat = initialize()  # ✅ Get the GroupChatManager
    tools = []

    role_responses = {}  # Dictionary to store responses grouped by role
    tool_names = []  # List to store tool names

    async for response in selector_group_chat.run_stream(task=messages_string):
        print(response)

        # Process text responses
        if isinstance(response, TextMessage) and response.content and response.source != "user":
            if response.source not in role_responses:
                role_responses[response.source] = []
            if response.content not in role_responses[response.source]:  # Avoid duplicates
                role_responses[response.source].append(response.content)

        # Process tool execution events
        if isinstance(response, ToolCallExecutionEvent):
            tools.append(response)
            tool_names.extend([
                content.name for content in response.content
                if hasattr(content, "name") and not content.name.startswith("transfer_to_")
            ])
            print("Extracted Tool Names:", tool_names)

    # Prepare formatted responses
    all_responses = ""
    for role, contents in role_responses.items():
        all_responses += f"**🔹 {role.capitalize()}**: {' '.join(contents)}\n\n"

    # Add tools used to the output if any
    # if tool_names:
    #     all_responses += f"**🛠 Tools Used:** {', '.join(tool_names)}"

    return all_responses
