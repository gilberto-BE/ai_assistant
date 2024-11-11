import dspy

ollama_port = 11434
ollama_url = f"http://localhost:{ollama_port}"
ollama_llm = dspy.LM(model="ollama/llama3.2:1b", api_base=ollama_url)
colbertv2_wiki_abstracts = dspy.LM(
    model="colbertv2-wiki-abstracts",
    url="http://20.102.90.50:2017/wiki17_abstracts",
)


dspy.settings.configure(lm=ollama_llm, rm=colbertv2_wiki_abstracts)

# dataset
##### TODO: issues with datasets
from datasets import load_dataset

dataset = load_dataset(
    "hotpot_qa", "distractor", split="train[:50] + validation[:50] + test[:50]"
)
# dataset = HotPotQA(train_seed=1, train_size=5, eval_seed=2023, dev_size=50, test_size=0)

print(dataset.train)

# Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
trainset = [x.with_inputs("question") for x in dataset.train]
devset = [x.with_inputs("question") for x in dataset.dev]

print(len(trainset), len(devset))
