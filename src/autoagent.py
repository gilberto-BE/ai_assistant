from autogen import AssistantAgent, UserProxyAgent
import os


class AIAgent:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        self.llm_config = {
            "model": self.model_name,
            "api_key": os.environ["OPENAI_API_KEY"],
        }
        self.assistant = AssistantAgent("assistant", llm_config=self.llm_config)
        self.user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)

    def start(self):
        self.user_proxy.initiate_chat(
            self.assistant,
            message="Tell me a joke about NVDA and TESLA stock prices.",
        )


if __name__ == "__main__":
    chat = AIAgent()
    chat.start()
