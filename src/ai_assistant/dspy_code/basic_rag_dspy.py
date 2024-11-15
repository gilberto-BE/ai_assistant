import os
import requests
import ujson
from dspy.evaluate import SemanticF1
import dspy
import torch 
import functools
from litellm import embedding as Embed

urls = [
    'https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_500.json',
    'https://huggingface.co/datasets/colbertv2/lotte_passages/resolve/main/technology/test_collection.jsonl',
    'https://huggingface.co/dspy/cache/resolve/main/index.pt'
]

class RAG(dspy.Module):
    def __init__(self, num_docs=5):
        self.num_docs = num_docs
        self.respond = dspy.ChainOfThought('context, question -> response')
        
    def forward(self, question):
        context = search(question, k=self.num_docs)
        return self.respond(context=context, question=question)

@functools.lru_cache(maxsize=None)
def search(query, k=5):
    query_embedding = torch.tensor(Embed(input=query, model='text-embedding-3-small').data[0]['embedding'])
    topk_scores, topk_indices = torch.matmul(index, query_embedding).topk(k)
    topK = [dict(score=score.item(), **corpus[idx]) for idx, score in zip(topk_indices, topk_scores)]
    return [doc['text'][:max_characters] for doc in topK]

if __name__ == "__main__":
    for url in urls:
        filename = os.path.basename(url)
        remote_size = int(requests.head(url, allow_redirects=True).headers.get('Content-Length',0))
        # print(f'Filename: {filename} \n filesize: {remote_size}\n\n')
        local_size = os.path.getsize(filename) if os.path.exists(filename) else 0
        
        if local_size != remote_size:
            print(f'Downloading: {filename}...')
            with requests.get(url, stream=True) as r, open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    print(f'Writing chunk of size: {len(chunk)}', sep='', end='\r', flush=True)
                    f.write(chunk)
    
    
    lm = dspy.LM(model='llama3.1')
    dspy.configure(lm=lm)
    
    with open('ragqa_arena_tech_500.json') as f:
        data = [dspy.Example(**d).with_inputs('question') for d in ujson.load(f)]
        trainset, valset,devset, testset = data[:50], data[50:150], data[150:300], data[300:500]
        
    metric = SemanticF1()
    evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=24, display_progress=True, display_table=3)
    
    with open('test_collection.jsonl') as f:
        corpus = [ujson.loads(line) for line in f]
        
    index = torch.load('index.pt', weights_only=True)
    max_characters = 4000
    rag = RAG()
    print(rag(question='what are high memory and low memory on linux?'))
    print(dspy.inspect_history())
    print(evaluate(RAG()))

    