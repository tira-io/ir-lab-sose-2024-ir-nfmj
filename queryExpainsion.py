from tira.third_party_integrations import ensure_pyterrier_is_loaded
from tira.rest_api_client import Client
import pyterrier as pt
import spacy 
import pandas as pd

nlp = spacy.load('en_core_web_md')

def get_similar_words(word, threshold=0.60):
    token = nlp(word)
    similar_words = []
    for vocab_word in nlp.vocab:
        if vocab_word.has_vector and vocab_word.is_lower and vocab_word.is_alpha:
            similarity = token.similarity(vocab_word)
            if similarity >= threshold:
                similar_words.append(vocab_word.text)
    return similar_words if similar_words else [word]

def get_best_word(original_word, similar_words, bm25, topic, pt_dataset):
    best_word = original_word
    best_score = -float('inf')
    
    for word in similar_words:
        topic_copy = topic.copy()
        topic_copy['query'] = topic_copy['query'].replace(original_word, word)
        
        print(f"Testing word: {word} in query: {topic_copy['query']}")
        
        experiment = pt.Experiment(
            [bm25],
            pd.DataFrame([topic_copy]),  
            pt_dataset.get_qrels(),
            ["ndcg_cut_10", "recip_rank", "recall_1000"],
            names=["BM25 - Low Entities"]
        )
        
        print(experiment)
        
        score = experiment[['ndcg_cut_10', 'recip_rank', 'recall_1000']].mean().mean()
        
        if score > best_score:
            best_score = score
            best_word = word
    
    return best_word

def queryExpansion(topics, bm25, pt_dataset):    
    expandedQueries = []
    originalQueries = topics['query'].tolist()

    for index, row in topics.iterrows():
        expandedTopic = []
        for word in row['query'].split(' '):
            similar_words = get_similar_words(word)
            best_word = get_best_word(word, similar_words, bm25, row, pt_dataset)
            expandedTopic.append(best_word)
        expandedQueries.append(' '.join(expandedTopic))
    topics['query'] = expandedQueries
    return topics, originalQueries, expandedQueries




ensure_pyterrier_is_loaded()
tira = Client()

pt_dataset = pt.get_dataset('irds:ir-lab-sose-2024/ir-acl-anthology-20240504-training')
topics = pt_dataset.get_topics(variant='title')

index = tira.pt.index('ir-lab-sose-2024/tira-ir-starter/Index (tira-ir-starter-pyterrier)', pt_dataset)
bm25 = pt.BatchRetrieve(index, wmodel="BM25")

expanded_topics, original_queries, expanded_queries = queryExpansion(topics, bm25, pt_dataset)


experiment = pt.Experiment(
    [bm25],
    expanded_topics,
    pt_dataset.get_qrels(),
    ["ndcg_cut_10", "recip_rank", "recall_1000"],
    names=["BM25 - Finaly Entities"]
)

for original, expanded in zip(original_queries, expanded_queries):
    print(f"Original Query: {original}")
    print(f"Expanded Query: {expanded}\n")

print(experiment)

