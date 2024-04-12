import tempfile
from autogen import ConversableAgent
from autogen.coding import LocalCommandLineCodeExecutor
import os


class CoderAgent:
    def __init__(self, use_llm: bool = False, model_name: str | None = None):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.executor = LocalCommandLineCodeExecutor(
            timeout=10,
            work_dir=self.temp_dir.name,
        )
        code_writer_system_message = """You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
Reply 'TERMINATE' in the end when everything is done.
"""
        self.model_name = model_name
        if self.model_name is not None:
            self.llm_config = {
                "config_list": [
                    {
                        "model": model_name,
                        "api_key": os.environ["OPENAI_API_KEY"],
                    }
                ]
            }
        else:
            self.llm_config = False
        self.code_executor_agent = ConversableAgent(
            "code_executor_agent",
            system_message=code_writer_system_message,
            llm_config=self.llm_config,
            code_execution_config=False,  # {"executor": self.executor},
            # human_input_mode="ALWAYS",
        )

    def reply(self, message):
        reply = self.code_executor_agent.generate_reply(
            messages=[{"role": "user", "content": message}]
        )
        print(reply)
        print(os.listdir(self.temp_dir.name))
        self.temp_dir.cleanup()

    def chat(self, message="Write Python code to calculate the 14th Fibonacci number."):
        chat_result = self.code_executor_agent.initiate_chat(
            self.code_executor_agent, message=message
        )
        print(chat_result)


if __name__ == "__main__":
    message = """This is a message with code block.
The code block is below:
```python
import numpy as np
import matplotlib.pyplot as plt
x = np.random.randint(0, 100, 100)
y = np.random.randint(0, 100, 100)
plt.scatter(x, y)
plt.savefig('scatter.png')
print('Scatter plot saved to scatter.png')
```
This is the end of the message.
    """

    agent = CoderAgent(model_name="gpt-3.5-turbo")
    # agent.reply(message=message)
    agent.chat()
