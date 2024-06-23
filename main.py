from tira.third_party_integrations import ensure_pyterrier_is_loaded, persist_and_normalize_run
from tira.rest_api_client import Client
import pyterrier as pt
import spacy

#Lade das spaCy NER-Modell
nlp = spacy.load("en_core_web_sm")

#Initialisiere die PyTerrier-Umgebung und den TIRA-Client
ensure_pyterrier_is_loaded()
tira = Client()

#Lade den Datensatz und den Index
pt_dataset = pt.get_dataset('irds:ir-lab-sose-2024/ir-acl-anthology-20240504-training')
index = tira.pt.index('ir-lab-sose-2024/tira-ir-starter/Index (tira-ir-starter-pyterrier)', pt_dataset)

bm25 = pt.BatchRetrieve(index, wmodel="BM25")

print('First, we have a short look at the first three topics:')

topics = pt_dataset.get_topics('text').head(3)
print(topics)

#Manuelle Erweiterung der Suchanfragen
manual_expansions = {
    'retrieval system improving effectiveness': 'retrieval system improving effectiveness search engines performance',
    'machine learning language identification': 'machine learning language identification natural language processing NLP',
    'social media detect self harm': 'social media detect self harm mental health online behavior'
}

def expand_query_with_manual_entities(query):
    return manual_expansions.get(query, query)

print('Now we do the retrieval...')

#Erweitere die Suchanfragen manuell
expanded_topics = topics.copy()
expanded_topics['query'] = topics['query'].apply(expand_query_with_manual_entities)

run = bm25(expanded_topics)

print('Done. Here are the first 10 entries of the run')
print(run.head(10))

persist_and_normalize_run(run, system_name='bm25-baseline', default_output='./runs')