import dspy

ollama_port = 11434 
ollama_url = f"http://localhost:{ollama_port}"
ollama_llm = dspy.LM(model="ollama/llama3.2:1b", api_base=ollama_url)
