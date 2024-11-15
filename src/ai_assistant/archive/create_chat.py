import dspy



def get_llm(name='openai/gpt-4o-mini'):
    lm = dspy.LM(name)
    dspy.configure(lm=lm)
    return lm

def get_ollama_llm():
    llm = dspy.OllamaLocal('phi3.5')
    dspy.configure(lm=llm)
    return llm


def get_hf_llm():
    llm = dspy.HFClientTGI(model="meta-llama/Llama-2-7b-hf", port=8080, url="http://localhost")
    dspy.configure(llm=llm)
    return llm

def question_and_answer(lm, question):
    dspy.configure(lm=lm)
    qa = dspy.Predict("question: str -> response: str")
    return qa(question="Who is the president of Sweden?").response

if __name__ == '__main__':
    print('start local llm:')
    lm = get_hf_llm()
    print(question_and_answer(lm, "Who is the president of Sweden?"))