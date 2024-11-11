from autogen import UserProxyAgent, ConversableAgent


local_llm_config = {
    "config_list": [
        {
            "model": "NotRequired",
            "api_key": "NotRequired",
            "base_url": "http://0.0.0.0:4000",
        }
    ],
    "cached_seed": None,
}

local_assistant = ConversableAgent("agent", llm_config=local_llm_config)
user_proxy = UserProxyAgent("user", code_execution_config=False)
assistant.initiate_chat(user_proxy, message="How can I help you today?")
