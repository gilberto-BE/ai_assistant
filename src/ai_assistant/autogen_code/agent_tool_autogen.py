from typing import Annotated, Literal
import os
from autogen import ConversableAgent, UserProxyAgent
from autogen import register_function


Operator = Literal["+", "-", "*", "/"]


def calculator(a: int, b: int, operator: Annotated[Operator, "operator"]) -> int:
    """
    Performs arithmetic operations on two numbers based on the given operator.

    Args:
        a (int): The first number.
        b (int): The second number.
        operator (str): The operator to perform the operation.
        Valid operators are '+', '-', '*', and '/'.

    Returns:
        int: The result of the arithmetic operation.

    Raises:
        ValueError: If an invalid operator is provided.

    """
    if operator == "+":
        return a + b
    elif operator == "-":
        return a - b
    elif operator == "*":
        return a * b
    elif operator == "/":
        return a / b
    else:
        raise ValueError("Invalid operator")


class AgentWithTools:
    def __init__(
        self,
        model_name: str = "gpt-4o",
        llm_config: dict | None = None,
    ):
        self.model_name = model_name
        code_writer_system_message = """You are a helpful AI assistant."""
        if llm_config is None and model_name is not None:
            self.llm_config = {
                "config_list": [
                    {
                        "model": self.model_name,
                        "api_key": os.environ["OPENAI_API_KEY"],
                    }
                ]
            }
        # elif llm_config is None and model_name is None:
        #     self.llm_config = {
        #         "config_list": [
        #             {
        #                 "model": "NotRequired",
        #                 "api_key": "NotRequired",
        #                 "base_url": "http://0.0.0.0:4000",
        #             }
        #         ],
        #         "cached_seed": None,
        #     }
        else:
            self.llm_config = llm_config
        self.start_assistant_agent()
        self.start_user_proxy_agent()

    def start_assistant_agent(self):
        self.assistant = ConversableAgent(
            name="Assistant",
            system_message="You are a helpful AI assistant. "
            "You can help with simple calculations."
            "Return 'TERMINATE' when the task is done.",
            llm_config=self.llm_config,
        )

    def start_user_proxy_agent(self):
        self.user_proxy = ConversableAgent(
            name="User",
            llm_config=False,
            is_termination_msg=lambda msg: msg.get("content") is not None
            and "TERMINATE" in msg["content"],
            human_input_mode="NEVER",
        )

    def register_tool(self, function_object: callable):
        name = function_object.__name__
        register_function(
            calculator,
            caller=self.assistant,  # The assistant agent can suggest calls to the calculator.
            executor=self.user_proxy,  # The user proxy agent can execute the calculator calls.
            name=name,
            description="A simple calculator",
        )

    def use_tool(self, message: str = "What is 5 + 3?"):
        return self.user_proxy.initiate_chat(self.assistant, message=message)


if __name__ == "__main__":
    agent = AgentWithTools(model_name="gpt-4o")
    agent.register_tool(calculator)
    chat_input = input("Enter a message: ")
    agent.use_tool(chat_input)
    print("TERMINATE")
