import os
import json
from pathlib import Path
import sys
import argparse
import re
import wandb
import torch
import requests
import pandas as pd
from transformers import (
        T5ForConditionalGeneration,
        T5Tokenizer,
        BartForConditionalGeneration,
        BartTokenizer
)
# Import utility functions
import sys
sys.path.insert(0, './utils/')
from utils import *

# Command line arguments
def parse(args):
    parser = argparse.ArgumentParser(description='evaluation')

    parser.add_argument("--model_name",type=str, default='t5-small', help='select model from: [t5-base, t5-small, bart-base, bart-large]', required=True)
    parser.add_argument("--embedding_name", type=str, default='transe', required=True, help="select embedding for entity ranker from: [transe, complex, distmult]")
    parser.add_argument("--input_dir",type=str, default='./', help='path to input directory', required=False)
    parser.add_argument("--output_dir", type=str, default='./eval_output/', required=False)
    parser.add_argument("--no_cuda", action="store_true", help="sets device to CPU", required=False)

    all_args = parser.parse_known_args(args)[0]
    return all_args

# Set up Wandb
def setup_wandb(args):
    config = dict(
            model_name=args.model_name,
            )
    wandb.init(
            config=config,
            name=args.model_name+"+"+args.embedding_name,
            project="dblp-kgqa-eval-runs",
            entity="dblp-kgqa",
            reinit=True
            )

def get_dataset(file_path):
    with Path(file_path).open() as json_file:
        data = json.load(json_file)
    return data

def get_gold_ents(test_data):
    gold_ents = []
    for item in test_data:
        gold_ents.append(set(item["entities"]))

    return gold_ents

# Load the pre-trained T5 model and tokenizer
def setup_model(model_name):
    if re.match(r"t5*", model_name):
        print("\nModel type: T5\n")
        model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict = True)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    else:
        print("\nModel type: BART\n")
        model = BartForConditionalGeneration.from_pretrained("facebook/"+model_name, return_dict = True)
        tokenizer = BartTokenizer.from_pretrained("facebook/"+model_name)

    output_dir = './models/'
    load_model(output_dir, model, 'model_{}.pth'.format(model_name))
    return model, tokenizer

def infer(question, model, tokenizer, device):
    # Tokenize
    input_question = tokenizer.encode_plus(question, padding=True, truncation=True, return_tensors='pt')
    input_question = input_question.to(device)
    # Generate answer containing label and type
    output = model.generate(input_ids=input_question['input_ids'], attention_mask=input_question['attention_mask'], max_length=512)
    predicted = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in output]
    return predicted

def separate_ents(predicted):
    ents = predicted.split("[SEP]")
    return ents[:-1]

def get_labels_types(ents):
    labels = []
    types =[]
    for ent in ents:
        all_types = []
        if ":" in ent:
            i = ent.rindex(":")
            labels.append(ent[:i].strip())
            temp = ent[i+1:]
            all_types = temp.split(',')
            all_types = [item.strip() for item in all_types[:-1]]
            types.append(all_types)
    return labels, types 

def predict_labels_types(question, model, tokenizer, device):
    predicted = infer(question, model, tokenizer, device)
    ents = separate_ents(predicted[0])
    labels, types = get_labels_types(ents)
    return labels, types 

def get_ranked_ent(question, embedding, labels, types):
    if embedding == 'transe':
        embed = 'transe'
    elif embedding == 'complex':
        embed = 'complex'
    elif embedding == 'distmult':
        embed = 'distmult'

    url = "http://ltcpu2:5001/entitylinker/" + embed
    headers = {"Content-Type": "application/json"}
    idx = 0
    responses = []
    resp_jsons = []
    for label, typs in zip(labels, types):
        print("\n:Label")
        print(label)
        data = {"question": question, "label": label, "type": typs}
        resp = requests.post(url, headers=headers, data=json.dumps(data))
        resp_json = resp.json()
        print("JSON Resp")
        print(resp_json)
        resp_jsons.append(resp_json)
        if len(resp_json):
            responses.append(resp_json[0][1][0])
        else:
            responses.append('')
    return resp_jsons, responses

def predict_rank1_ents(args, test_data, model, tokenizer, device):
    predicted = []
    results = []
    for item in test_data:
        question = item["question"]
        labels, types = predict_labels_types(question, model, tokenizer, device)
        resp_jsons, response = get_ranked_ent(question, args.embedding_name, labels, types)
        predicted.append(set(response))
        results.append(resp_jsons)
    return results, predicted

def calc_metrics(gold_entities, predicted_entities):
    tp, fp, fn, tn = 0., 0., 0., 0.
    label = 0
    for gold_ents, pred_ents in zip(gold_entities, predicted_entities):
        print("\nGold entities")
        print(gold_ents)
        print("Predicted entities")
        print(pred_ents)
        for pred_ent in pred_ents:
            if pred_ent in gold_ents:
                tp += 1
            else:
                fp += 1
        for gold_ent in gold_ents:
            if gold_ent not in pred_ents:
                fn += 1
        label += 1
    if (tp + fp) != 0:
        precision = tp / (tp + fp) # out of predicted entities how many are the actual entities also
    else:
        precision = 0
    if (tp + fn) != 0:
        recall = tp / (tp + fn) # out of actual entities how many were predicted by the model
    else:
        recall = 0
    if (precision + recall) != 0:
        f1 = (2.0 * precision * recall) / (precision + recall)
    else: 
        f1 = 0
    return precision, recall, f1

def calc_mrr(gold_entities, results):
    mrr = 0
    total = 0
    for ques, gold_ents in enumerate(gold_entities):
        for gold_ent in gold_ents:
            found = 0
            total += 1
            for i, resp_jsons in enumerate(results[ques]):
                for rank, item in enumerate(resp_jsons):
                    if gold_ent == item[1][0]:
                        mrr += 1.0/(rank+1)
                        found = 1
                        break
                if found:
                    break
            if found:
                break
    return mrr / total

if __name__ == '__main__':
    args = parse(sys.argv[1:])
    # Get test dataset
    file_path = os.path.join(args.input_dir, "test_data.json")
    test_data = get_dataset(file_path)
    # Actual entities
    gold_entities = get_gold_ents(test_data)
    # Load model and tokenizer
    model, tokenizer = setup_model(args.model_name)
    # Set device
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else 'cpu')
    model = model.to(device)
    # Set up wandb
    setup_wandb(args)
    # Predicted entities
    results, predicted_entities = predict_rank1_ents(args, test_data, model, tokenizer, device)
    # Calculate metrics
    precision, recall, f1 = calc_metrics(gold_entities, predicted_entities)
    print("\nPrecision: " + str(precision))
    print("\nRecall: " + str(recall))
    print("\nF1: " + str(f1))
    wandb.log({"Precision":precision, "Recall":recall, "F1":f1})
    # Calculate Mean Reciprocal Rank (MRR)
    mrr = calc_mrr(gold_entities, results)
    print("\nMRR: " + str(mrr))
    wandb.log({"MRR": mrr})
    df = pd.DataFrame({"model_name":args.model_name, "Precision": precision,"Recall": recall, "F1": f1, "MRR": mrr}, index=[0])
    df.to_json(os.path.join(args.output_dir, "{}_test_results.json".format(args.model_name)), orient = 'records', compression = 'infer', index = 'false')
