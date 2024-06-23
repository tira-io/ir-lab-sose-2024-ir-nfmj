from tira.third_party_integrations import ensure_pyterrier_is_loaded, persist_and_normalize_run
from tira.rest_api_client import Client
import pyterrier as pt
import pandas as pd

# Laden der benötigten Bibliotheken
ensure_pyterrier_is_loaded()
tira = Client()

# Laden des Datasets
pt_dataset = pt.get_dataset('irds:ir-lab-sose-2024/ir-acl-anthology-20240504-training')

# Abrufen der Themen (Queries)
topics = pt_dataset.get_topics(variant='title')

# Anwenden der Entity-Linking Transformation auf die Queries
query_entity_linking = tira.pt.transform_queries('ir-benchmarks/marcel-gohsen/entity-linking', pt_dataset)
linked_queries = query_entity_linking(topics)

# Initialisieren der Listen für low und high Entities
lowEntity = []
highEntity = []

# Kategorisieren der Queries basierend auf der Anzahl der Entities
for i in range(len(linked_queries)):
    entities = linked_queries.iloc[i].to_dict().get('entities')
    if entities is not None and len(entities) < 10:
        lowEntity.append(linked_queries.iloc[i])
    elif entities is not None:
        highEntity.append(linked_queries.iloc[i])

# Funktion zur Entfernung von Fragezeichen aus den Queries
def remove_question_marks(df):
    df['query'] = df['query'].str.replace('?', '', regex=False)
    return df

# Konvertieren der lowEntity und highEntity Listen in DataFrames
lowEntity_df = pd.DataFrame(lowEntity)
highEntity_df = pd.DataFrame(highEntity)

# Entfernen von Fragezeichen aus den Queries
lowEntity_df = remove_question_marks(lowEntity_df)
highEntity_df = remove_question_marks(highEntity_df)

# Extrahieren der relevanten Topics für lowEntity und highEntity
lowEntity_topics = lowEntity_df[['qid', 'query']]
highEntity_topics = highEntity_df[['qid', 'query']]

# Index laden
index = tira.pt.index('ir-lab-sose-2024/tira-ir-starter/Index (tira-ir-starter-pyterrier)', pt_dataset)
bm25 = pt.BatchRetrieve(index, wmodel="BM25")

# Auswertung der lowEntities
experiment_lowEntity = pt.Experiment(
    [bm25],
    lowEntity_topics,
    pt_dataset.get_qrels(),
    ["ndcg_cut.10", "recip_rank", "recall_1000"],
    names=["BM25 - Low Entities"]
)

# Ausgabe der Ergebnisse für lowEntities
print("Ergebnisse für Low Entities:")
print(experiment_lowEntity)

# Auswertung der highEntities
experiment_highEntity = pt.Experiment(
    [bm25],
    highEntity_topics,
    pt_dataset.get_qrels(),
    ["ndcg_cut.10", "recip_rank", "recall_1000"],
    names=["BM25 - High Entities"]
)

# Ausgabe der Ergebnisse für High Entities
print("Ergebnisse für High Entities:")
print(experiment_highEntity)
