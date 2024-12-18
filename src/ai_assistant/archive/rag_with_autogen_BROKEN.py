import autogen
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
import os


class RAGAgent:
    def __init__(
        self,
        # llm_config,
        model_name="gpt-3.5-turbo",
        docs_path="https://raw.githubusercontent.com/microsoft/autogen/main/README.md",
    ):
        self.llm_config = {
            "timeout": 600,
            "cache_seed": 42,
            "config_list": [
                {
                    "model": model_name,
                    "api_key": os.environ["OPENAI_API_KEY"],
                }
            ],
        }
        # self.llm_config = llm_config
        self.docs_path = docs_path
        self.start_assistant()

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

    def start_chat(self):

        self.assistant.reset()
        self.ragproxyagent.initiate_chat(
            self.assistant,
            message=self.ragproxyagent.message_generator,
            problem="What is the capital of kenya?",
        )

        # self.userproxyagent = autogen.UserProxyAgent(name="userproxyagent")
        # self.userproxyagent.initiate_chat(self.assistant, message="what is autogen?")


if __name__ == "__main__":
    agent = RAGAgent()
    agent.start_chat()
