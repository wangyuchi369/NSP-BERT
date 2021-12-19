from sentence_transformers import SentenceTransformer
import scipy
import json_lines
from tqdm import tqdm
import scipy
import numpy as np
model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
#%%
with open('dev.jsonl', 'r') as f:
    data = [item for item in json_lines.reader(f)]
num = len(data)
mapping = {0:'A',
           1:'B',
           2:'C',
           3:'D',
           4:'E'}
count = 0
for i in tqdm(range(num)):
    ground_truth = data[i]['answerKey']
    question_answer = data[i]['question']
    sentences = [item['text'] for item in question_answer['choices']]
    sentence_embeddings = model.encode(sentences)

    query = data[i]['question']['stem']  #  A query sentence uses for searching semantic similarity score.
    queries = [query]
    query_embeddings = model.encode(queries)

    print("Semantic Search Results")
    number_top_matches = 3
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]
        idx = np.argmin(distances)
        print(idx)
        print('预测的为',sentences[idx])
        if mapping[idx] == ground_truth:
            count+=1
            print('true')
        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])
        print("Query:", query)
        print("Top {} most similar sentences in corpus:".format(number_top_matches))

        for idx, distance in results[0:number_top_matches]:
            print(sentences[idx].strip(), "(Cosine Score: %.4f)" % (1-distance))
        print('\n')
print('accuracy is',count/num)
