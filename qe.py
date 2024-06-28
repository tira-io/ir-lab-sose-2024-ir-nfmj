# Imports
from tira.third_party_integrations import ensure_pyterrier_is_loaded, persist_and_normalize_run
from tira.rest_api_client import Client
import pyterrier as pt

def custom_stopwords():
        # Der Pfad zur Textdatei
    file_path = './stopwords/stopword-list.txt' # Default Stopword List by Terrier

    # Initialisiere eine leere Liste
    stopwords_list = []

    # Öffne die Datei und lese jede Zeile
    with open(file_path, 'r') as file:
        for line in file:
            # Entferne führende und nachfolgende Leerzeichen (einschließlich neuer Zeilen)
            stripped_line = line.strip()
            if "information" in stripped_line: # Ignoriere Stopwort Information
                continue 
            # Füge die bereinigte Zeile zur Liste hinzu, falls sie nicht leer ist
            if stripped_line:
                stopwords_list.append(stripped_line)
    return stopwords_list

# Create a REST client to the TIRA platform for retrieving the pre-indexed data.
ensure_pyterrier_is_loaded()
tira = Client()

pt_dataset = pt.get_dataset('irds:ir-lab-sose-2024/ir-acl-anthology-20240504-training')

iter_indexer = pt.IterDictIndexer("./index", stopwords=custom_stopwords(),meta={'docno': 50, 'text': 4096}, overwrite=True, blocks=True)

index = iter_indexer.index(pt_dataset.get_corpus_iter())

# The dataset: the union of the IR Anthology and the ACL Anthology
# This line creates an IRDSDataset object and registers it under the name provided as an argument.


topics = pt_dataset.get_topics(variant='title')

query_entity_linking = tira.pt.transform_queries('ir-benchmarks/marcel-gohsen/entity-linking', pt_dataset)

linked_queries = query_entity_linking(topics)

bm25 = pt.BatchRetrieve("./index", wmodel="BM25")

# Fragezeichen von den Queries entfernen, sonst gibt es Probleme bei der Transformation
for index, row in linked_queries.iterrows():
    if '?' in str(row['query']):
        linked_queries['query'] = linked_queries['query'].str.replace('?', '')

def keyphrase_containment_checker(input_phrase, wanted_string):
    # Überprüfen, ob der gewollte String bereits in der Input Phrase enthalten ist
    if wanted_string in input_phrase:
        return True
    else:
        return False
    
def entity_keyphrase_length_checker(entity_keyphrase):
    # String nach Leerzeichen aufsplitten
    entity_word = entity_keyphrase.split(" ")
    # Anzahl der Elemente ermitteln
    return len(entity_word)

def add_missing_terms(input_phrase, wanted_string):
    wanted_string = wanted_string.lower()
    if not keyphrase_containment_checker(input_phrase, wanted_string):
        input_phrase += wanted_string + f" "
        return input_phrase
    else:
        return input_phrase


def query_rewrite(linked_queries, score_threshold=0.9, entity_list_size=2):
    queries_entities = linked_queries['entities'].to_dict()
    for qid in queries_entities:
        query = linked_queries['query'][qid];
        query_entities = queries_entities[qid] # Eine Liste mit entitäten, die von jeder Query kommt
        if(len(query_entities) > 0): # Manche Query besitzen vielleicht keine Entitäten
            keyphrases = {} # Dict für die Keyphrases

            j = 0;
            for i in range(0,entity_list_size):
                if(entity_list_size < len(query_entities)):
                    entity_keyphrase = query_entities[i]['mention']
                    entity_score = query_entities[i]['score']
                    keyphrases[i] = (entity_keyphrase, entity_score) # keyphrases[i][0] := Entität und keyphrases[i][1] := Der Score der Entität
                else:                                                # Wenn entity_list_size zu groß ist für len(query_entities), dann nehme die maximale mögliche anzahl, also len(query_entities)
                    entity_keyphrase = query_entities[i]['mention']
                    entity_score = query_entities[i]['score']
                    keyphrases[i] = (entity_keyphrase, entity_score)
                    j += 1
                    if(j == len(query_entities)):
                        break
            

            delta = 0 # Mittelwert des Scores von den einzelen Entitäten aus einer Query
            for i in range(0,len(keyphrases)):
                delta += keyphrases[i][1]
            delta = delta / len(keyphrases)
            
            result = ""
            old_result = ""
            if( delta >= score_threshold):
                for i in range(0,len(keyphrases)):
                    phrase = keyphrases[i][0]
                    if entity_keyphrase_length_checker(phrase) >= 2 and not keyphrase_containment_checker(old_result, phrase):
                        result += f'"{phrase}"' + f" "
                        old_result = result
                
                if len(result) != 0:
                    query_word_list = query.split(" ")
                    for i in range(0, len(query_word_list)):
                        result = add_missing_terms(result, query_word_list[i])
                    result = result.strip(" ");
                    print("Changing Query from " +  '[' + linked_queries['query'][qid] + ']' + " to [" + result + "]")
                    linked_queries['query'][qid] = result

query_rewrite(linked_queries, 0.9, 2)

bm25QR = linked_queries >> bm25 

print('Now we do the retrieval...')
runDefault = bm25(pt_dataset.get_topics('text'))
runQR = bm25QR(pt_dataset.get_topics('text'))
print("Done!")

persist_and_normalize_run(runDefault,  system_name='bm25-baseline', default_output='./runs/defaultRuns')
persist_and_normalize_run(runQR, system_name='bm25-baseline', default_output='./runs/QRRuns')

