import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch


train = [('Who was the director of the 2009 movie featuring Peter Outerbridge as William Easton?', 'Kevin Greutert'),
         ('The heir to the Du Pont family fortune sponsored what wrestling team?', 'Foxcatcher'),
         ('In what year was the star of To Hell and Back born?', '1925'),
         ('Which award did the first book of Gary Zukav receive?', 'U.S. National Book Award'),
         ('What documentary about the Gilgo Beach Killer debuted on A&E?', 'The Killing Season'),
         ('Which author is English: John Braine or Studs Terkel?', 'John Braine'),
         ('Who produced the album that included a re-recording of "Lithium"?', 'Butch Vig')]

train = [dspy.Example(question=question, answer=answer).with_inputs('question') for question, answer in train]
dev = [('Who has a broader scope of profession: E. L. Doctorow or Julia Peterkin?', 'E. L. Doctorow'),
       ('Right Back At It Again contains lyrics co-written by the singer born in what city?', 'Gainesville, Florida'),
       ('What year was the party of the winner of the 1971 San Francisco mayoral election founded?', '1828'),
       ('Anthony Dirrell is the brother of which super middleweight title holder?', 'Andre Dirrell'),
       ('The sports nutrition business established by Oliver Cookson is based in which county in the UK?', 'Cheshire'),
       ('Find the birth date of the actor who played roles in First Wives Club and Searching for the Elephant.', 'February 13, 1980'),
       ('Kyle Moran was born in the town on what river?', 'Castletown River'),
       ("The actress who played the niece in the Priest film was born in what city, country?", 'Surrey, England'),
       ('Name the movie in which the daughter of Noel Harrison plays Violet Trefusis.', 'Portrait of a Marriage'),
       ('What year was the father of the Princes in the Tower born?', '1442'),
       ('What river is near the Crichton Collegiate Church?', 'the River Tyne'),
       ('Who purchased the team Michael Schumacher raced for in the 1995 Monaco Grand Prix in 2000?', 'Renault'),
       ('André Zucca was a French photographer who worked with a German propaganda magazine published by what Nazi organization?', 'the Wehrmacht')]

dev = [dspy.Example(question=question, answer=answer).with_inputs('question') for question, answer in dev]

def run_local_llm():
    llm = dspy.HFClientTGI(model="meta-llama/Llama-2-13b-chat-hf", port=[7140, 7141, 7142, 7143], url="http://localhost",)
    colbertv2 = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
    dspy.settings.configure(llm=llm, colbertv2=colbertv2)
    predict = dspy.Predict('question -> answer')
    print(predict(question='What is the capital of Germany?'))
    
class CoT(dspy.Module):
    def __init__(self):
        self.generate_answer = dspy.ChainOfThought('question -> answer')
        
    def forward(self, question):
        return self.generate_answer(question=question)
    

    
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
    
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)
    
from dspy.teleprompt import BootstrapFewShot

# Validation logic: check that the predicted answer is correct.
# Also check that the retrieved context does actually contain that answer.
def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM



if __name__ == '__main__':
    import dspy

    turbo = dspy.OpenAI(model='gpt-3.5-turbo')
    colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

    dspy.settings.configure(lm=turbo, rm=colbertv2_wiki17_abstracts)
    
    from dspy.datasets import HotPotQA

    # Load the dataset.
    dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)

    # Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
    trainset = [x.with_inputs('question') for x in dataset.train]
    devset = [x.with_inputs('question') for x in dataset.dev]

    print(len(trainset), len(devset))
    
    # Set up a basic teleprompter, which will compile our RAG program.
    teleprompter = BootstrapFewShot(metric=validate_context_and_answer)

    # Compile!
    compiled_rag = teleprompter.compile(RAG(), trainset=trainset)
    
    # Ask any question you like to this simple RAG program.
    my_question = "What castle did David Gregory inherit?"

    # Get the prediction. This contains `pred.context` and `pred.answer`.
    pred = compiled_rag(my_question)

    # Print the contexts and the answer.
    print(f"Question: {my_question}")
    print(f"Predicted Answer: {pred.answer}")
    print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")