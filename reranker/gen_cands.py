import sys,os,json
from elasticsearch import Elasticsearch
from fuzzywuzzy import fuzz
import random
import requests


es = Elasticsearch("http://ltcpu2:2200")

def label_search_es(label, enttype):
    try:
        resp = es.search(index="dblplabelsindex02", query={"bool": {"must": [{"match": {"label": label}}],
                                "filter": {"terms": {"types": enttype}}}})

        entities = []
        for source in resp['hits']['hits']:
            entities.append([source['_source']['entity'], source['_source']['label'].replace('"','')])
        return entities
    except Exception as err:
        print(err)
        return []

def gettype(entid):
    try:
        url = 'http://ltcpu2:8897/sparql'
        query = '''
                     SELECT distinct ?x where { 
                                                 %s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?x .
                     } 
                '''%(entid)
        #print(query)
        headers = {'Accept':'application/sparql-results+json'}
        r = requests.get(url, headers=headers, params={'format': 'json', 'query': query})
        json_format = r.json()
        #print(json_format)
        results =[x['x']['value'] for x in json_format['results']['bindings']]
        return results
    except Exception as err:
        print(err)
        return ''

def fetchembedding(entid):
    try:
        resp = es.search(index="dblpembedsdistmultindex01", query={"match":{"key":entid}})
        #print(resp)
        embedding = resp['hits']['hits'][0]['_source']['embedding']
        return embedding
    except Exception as err:
        print(err)
        return []

def getlabels(entid):
    try:
        resp = es.search(index="dblplabelsindex02", query={"match":{"entity":entid}})
        #print(resp)
        labels = []
        for source in resp['hits']['hits']:
            labels.append(source['_source']['label'].replace('"',''))
        return labels
    except Exception as err:
        print("here",err)
        return []

def remove_all_occurrences(lst, item):
    while item in lst:
        lst.remove(item)

d = json.loads(open(sys.argv[1]).read())

f = open(sys.argv[2],'w')
for item in d['questions']:
    try:
        entities = item['entities']
        #print(item['id'])
        #print(item['question']['string'])
        #print(entities)
        newents = []
        for ent in entities:
            res = getlabels(ent)
            enttypes = gettype(ent)
            entlabel = res[0]
            print("res:",entlabel)
            cands = label_search_es(entlabel, enttypes)
            goldfuzz = fuzz.token_set_ratio(entlabel, item["question"])
            remove_all_occurrences(cands,[ent,entlabel])
            negative_sample = random.choice(cands)
            #print("gold ent:",ent)
            #print("neg  ent:",negative_sample[0])
            goldemb = fetchembedding(ent)
            negemb = fetchembedding(negative_sample[0])
            neglabel  = negative_sample[1]
            negfuzz = fuzz.token_set_ratio(neglabel, item["question"])
            newents.append({'goldent':ent, 'goldemb':goldemb, 'goldlabel':entlabel, "goldfuzz": goldfuzz, 'negent':negative_sample[0], 'negemb':negemb , 'neglabel':neglabel, "negfuzz":negfuzz})
        newitem = {'id':item['id'], 'question':item['question'], 'entity_samples': newents}
        #print(newitem)
        f.write(json.dumps(newitem)+'\n')
    except Exception as err:
        print(err)
f.close()
