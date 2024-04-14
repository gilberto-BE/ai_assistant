from autogen import UserProxyAgent, ConversableAgent


class LocalAgent:
    def __init__(self):
        self.local_llm_config = {
            "config_list": [
                {
                    "model": "NotReuired",
                    "api_key": "NotRequired",
                    "base_url": "http://0.0.0.0:4000",
                }
            ],
            "cache_seed": None,
        }

    def start_chat(self):
        self.assistant = ConversableAgent("agent", llm_config=self.local_llm_config)
        self.user_proxy = UserProxyAgent("user", code_execution_config=False)
        self.assistant.initiate_chat(
            self.user_proxy, message="How can I help you today?"
        )


if __name__ == "__main__":
    agent = LocalAgent()
    agent.start_chat()
