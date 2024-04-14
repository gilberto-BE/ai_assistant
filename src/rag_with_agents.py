import autogen
from autogen.agentchat.contrib.retrieve_assisnatant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent


class RAGAgent:
    def __init__(
        self,
        llm_config,
        docs_path="https://raw.githubusercontent.com/microsoft/autogen/main/README.md",
    ):
        self.llm_config = llm_config
        self.docs_path = docs_path

    def start_assistant(self):
        self.assistant = RetrieveAssistantAgent(
            name="assistant",
            system_message="You are a helpful assistant.",
            llm_config=self.llm_config,
        )
        self.ragproxyagent = RetrieveUserProxyAgent(
            name="ragproxyagent",
            retrieve_config={
                "task": "qa",
                "docs_path": self.docs_path,
            },
        )

    def init_chat(self):
        self.assistant.reset
