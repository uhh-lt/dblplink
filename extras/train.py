# Import utility functions
import sys
sys.path.insert(0, './../utils/')
from utils import *

# Import modules
import torch
import argparse
import glob
import os
import json
import time
import logging
import random
import re
from string import punctuation
import wandb

import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import textwrap
from datetime import datetime

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
    BartForConditionalGeneration,
    BartTokenizer
)

# Using rich for displaying on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console

# Defining rich console logger
console = Console(record=True)

# Training logger to log training progress
training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Training Loss", justify="center"),
        title="Training Logs",
        pad_edge=False,
        box=box.ASCII,
        )
# Validation logger to log validation progress
validation_logger = Table(
        Column("Epoch", justify="center"),
        Column("Validation Loss", justify="center"),
        title="Validation Logs",
        pad_edge=False,
        box=box.ASCII,
        )
        
# Command line arguments
def parse(args):
    parser = argparse.ArgumentParser(description='dblp-kgqa-getlabels')

    parser.add_argument("--train", action="store_false", help='whether to perform training or not', required=False)
    parser.add_argument("--test", action="store_false", help='whether to perform testing or not', required=False)
    parser.add_argument("--model_name",type=str, default='t5-base', help='select model from: [t5-base, t5-small, t5-large, bart-large-cnn]', required=False)
    parser.add_argument("--input_dir",type=str, default='./', help='path to input directory', required=False)
    parser.add_argument("--batch_size", type=int, default=8, help='Batch size', required=False)
    parser.add_argument("--epochs", type=int, default=10, help='Number of epochs', required=False)
    parser.add_argument("--no_cuda", action="store_true", help="sets device to CPU", required=False)
    parser.add_argument("--seed", type=int, default=7, required=False)
    parser.add_argument("--lr", type=float, default=0.0001, required=False)
    parser.add_argument("--output_dir", type=str, default='./output/', required=False)

    all_args = parser.parse_known_args(args)[0]
    return all_args

# Define custom dataset class
class DBLPDataset(Dataset):
    def __init__(self, file_path):
        with Path(file_path).open() as json_file:
            self.data = json.load(json_file)

        self.questions = []
        self.answers = []
        for idx, item in enumerate(self.data):
            answer = ""
            for i, label in enumerate(item["labels"]):
                if label is None:
                    answer += "  : "
                    answer += " , "
                    answer += "[SEP] "
                else:
                    answer += label + " : "
                    for j, typ in enumerate(item["types"][i]):
                        answer += typ + ", "
                    answer += "[SEP] "
            self.answers.append(answer)
            self.questions.append(item["question"])
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return self.data[index]

# Load the pre-trained T5 model and tokenizer
def setup_train(args):
    model_name = args.model_name
    batch_size = args.batch_size
    args.output_dir = './output/'+model_name+"_bs="+str(batch_size)+'/'
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
    if re.match(r"t5*", model_name):
        print("\nModel type: T5\n")
        model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict = True)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    else:
        print("\nModel type: BART\n")
        model = BartForConditionalGeneration.from_pretrained("facebook/"+model_name, return_dict = True)
        tokenizer = BartTokenizer.from_pretrained("facebook/"+model_name)
    return model, tokenizer

# Create the dataset and data loader
def create_dataloader(args):
    train_dataset = DBLPDataset(os.path.join(args.input_dir, "train_data.json"))
    valid_dataset = DBLPDataset(os.path.join(args.input_dir, "valid_data.json"))
    train_dataloader = DataLoader(list(zip(train_dataset.questions, train_dataset.answers)), batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(list(zip(valid_dataset.questions, valid_dataset.answers)), batch_size=args.batch_size, shuffle=True)
    return train_dataloader, valid_dataloader

# Set up Wandb
def setup_wandb(args):
    config = dict(
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
            )
    wandb.init(
            config=config,
            name=args.model_name+"_bs="+str(args.batch_size)+"_e="+str(args.epochs)+"_lr="+str(args.lr),
            project="dblp-kgqa-runs",
            entity="dblp-kgqa",
            reinit=True
            )

# Define training procedure
def train(epoch, model, optimizer, train_dataloader, device):
    model.train()
    train_loss = 0
    
    for idx, batch in enumerate(train_dataloader):
        print("batch = " + str(idx))
        inputs = tokenizer.batch_encode_plus(batch[0], padding=True, truncation=True, return_tensors='pt')
        inputs = {key: val.to(device) for key, val in inputs.items()}

        labels = tokenizer.batch_encode_plus(batch[1], padding=True, truncation=True, return_tensors='pt')
        labels['input_ids'][labels['input_ids'] == 0] = -100
        labels = {key: val.to(device) for key, val in labels.items()}

        optimizer.zero_grad()
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                        labels=labels['input_ids'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    loss = train_loss/len(train_dataloader)
    training_logger.add_row(str(epoch), str(loss))
    wandb.log({"epoch": epoch, "train_loss": loss})
    print(f'Train Loss: {loss:.3f}')

# Define validation procedure
def valid(epoch, model, optimizer, valid_dataloader, device, batch_size):
    model.eval()
    val_loss = 0
    outputs = []
    targets = []
    exact_matches_predicted = 0
    total_predicted = 0

    for idx, batch in enumerate(valid_dataloader):
        print("batch = " + str(idx))
        inputs = tokenizer.batch_encode_plus(batch[0], padding=True, truncation=True, return_tensors='pt')
        inputs = {key: val.to(device) for key, val in inputs.items()}

        labels = tokenizer.batch_encode_plus(batch[1], padding=True, truncation=True, return_tensors='pt')
        labels = {key: val.to(device) for key, val in labels.items()}

        optimizer.zero_grad()
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=labels['input_ids'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        val_loss += loss.item()

        outs = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=512)
        predicted = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in outs]
        questions = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in inputs['input_ids']]
        targets = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in labels['input_ids']]

        exact_matches_predicted += exact_matches(targets, predicted)
        total_predicted += batch_size

    # Print two random samples for this epoch
    indices = random.sample(range(len(questions)), 2)
    for index in indices:
        print("\nQuestion:\n{}".format(questions[index]))
        print("\nTarget\n{}".format(targets[index]))
        print("\nPredicted:\n{}".format(predicted[index]))
    
    loss = val_loss/len(valid_dataloader)
    validation_logger.add_row(str(epoch), str(loss))
    print(f'Validation Loss: {loss:.3f}')
    print("\nExact Matches = {}/{}\n".format(exact_matches_predicted, total_predicted))
    print("\nValidation Accuracy = {}\n".format(exact_matches_predicted/total_predicted))
    wandb.log({"epoch":epoch, "val_loss":loss})
    wandb.log({"epoch":epoch, "val_accuracy": exact_matches_predicted/total_predicted})
    return loss

# Define test
def test(model, test_dataloader, device):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    exact_matches_predicted = 0
    total_predicted = 0
    with torch.no_grad():
        for batch in test_dataloader:
            inputs = tokenizer.batch_encode_plus(batch[0], padding=True, truncation=True, return_tensors='pt')
            inputs = {key: val.to(device) for key, val in inputs.items()}

            labels = tokenizer.batch_encode_plus(batch[1], padding=True, truncation=True, return_tensors='pt')
            labels = {key: val.to(device) for key, val in labels.items()}
            
            outs = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=512)
            predicted = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in outs]
            questions = [tokenizer.decode(ids) for ids in inputs['input_ids']]
            targets = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in labels['input_ids']]

            exact_matches_predicted += exact_matches(targets, predicted)
            total_predicted += batch_size
 
    print("\nExact Match Test Accuracy = {}\n".format(exact_matches_predicted/total_predicted))
    wandb.log({"test_accuracy": exact_matches_predicted/total_predicted})

# Fine tune
if __name__ == '__main__':
    args = parse(sys.argv[1:])
    epochs = args.epochs
    model_name = args.model_name
    print("Model: {}".format(model_name))
    if args.train:
        # Create model
        model, tokenizer = setup_train(args)
        # Define optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        # Create dataloader
        train_dataloader, valid_dataloader = create_dataloader(args)
        # Set device
        cuda = not args.no_cuda and torch.cuda.is_available()
        kwargs = {'num_workers': 2, 'pin_memory': True} if cuda else {}
        device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
        model = model.to(device)
        print(device)
        # Set up wandb
        setup_wandb(args)
        wandb.watch(model, log_freq=10)

        for epoch in range(epochs):
            print("\nEpoch: {}/{}".format(epoch+1, epochs))
            train(epoch, model, optimizer, train_dataloader, device)
            val_loss = valid(epoch, model, optimizer, valid_dataloader, device, args.batch_size)

            # To save model
            if epoch == 0:
                current_best = val_loss
                save_model(args.output_dir, model, model_name,  epoch, val_loss)
            else:
                if val_loss < current_best:
                    current_best = val_loss
                    save_model(args.output_dir, model, model_name,  epoch, val_loss)
        # Print predictions and losses
        console.print(training_logger)
        console.print(validation_logger)
        now = datetime.now()
        log_name = "logs_{}.txt".format(now.strftime("%d/%m/%Y_%H:%M:%S"))
        file_path = os.path.join(args.output_dir, log_name) 
        f = open(file_path, "w")
        console.save_text(file_path)    
        wandb.finish() 
