from agent_wrapper import AgentWrapper
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from agent_wrapper import AgentWrapper

from autogen_agentchat.teams import Swarm
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from typing import Sequence
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.messages import ToolCallExecutionEvent

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

    selector_prompt = """
    Select an agent to perform task.

    {roles}

    Current conversation context:
    {history}

    Read the above conversation, then select an agent from {participants} to perform the next task.
    After an agent responds, return the response, and wait for input from user.
    Only select one agent and stop the conversation strictly and wait for user input.
    """
   
    selector_prompt_3= """
    Given a conversation history string, extract the last message and determine the next speaker.

    :param conversation_history: A multi-line string containing the chat history.
    :return: A tuple (last_message, next_speaker)

    Example:
    message: [TextMessage(source='user', models_usage=None, metadata={}, content='User: hi\nAssistant: WorkloadsAssistant:How can I assist you in designing your Azure workload today?\nUser: design a workload)]
    reponse: speaker selection based on "design a workload"
    """

    termination = HandoffTermination(target="user") | TextMentionTermination("TERMINATE") | (MaxMessageTermination(max_messages=6))

    swarm_teams_manager = Swarm(
        participants = agents,
        # model_client=AgentWrapper.get_model_client(),
        termination_condition=termination,
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
