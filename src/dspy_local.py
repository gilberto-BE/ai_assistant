import dspy

def init_llm():
    lm = dspy.LM('openai/gpt-4o-mini')
    dspy.configure(lm=lm)
    return lm

def question_and_answer(lm, question):
    dspy.configure(lm=lm)
    qa = dspy.Predict("question: str -> response: str")
    return qa(question="Who is the president of Sweden?")

if __name__ == '__main__':
    print('start local llm:')
    lm = init_llm()
    print(question_and_answer(lm, "Who is the president of Sweden?"))