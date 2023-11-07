# Import modules
import os
import torch
from tqdm import tqdm
import transformers

def save_model(output_dir, model, model_name, epoch, score):
    model_file = os.path.join(output_dir, 'model_{}_epoch{}_score{:.4f}.pth'.format(model_name, epoch, score))
    torch.save(model.state_dict(), model_file)

def load_model(output_dir, model, model_name):
    model_file = os.path.join(output_dir, model_name)
    assert os.path.isfile(model_file), 'Error: no model found!'
    model_state = torch.load(model_file, map_location=torch.device('cpu'))
    model.load_state_dict(model_state)

def exact_matches(targets, predictions):
    exact_matches = 0

    for target, pred in zip(targets, predictions):
        if target == pred:
            exact_matches += 1
    return exact_matches

def eval_exact_match(predictions, gold_answers, tokenizer):
    exact_match = 0

    for ground_truths, prediction in tqdm(zip(gold_answers, predictions)):
        # Remove pad tokens
        tokens_to_remove = {
                tokenizer.pad_token_id,
                tokenizer.eos_token_id,
                tokenizer.bos_token_id,
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
                tokenizer.mask_token_id
                }
        prediction = list(filter(lambda token: token not in tokens_to_remove, prediction))
        ground_truths = list(filter(lambda token: token not in tokens_to_remove, ground_truths))
        exact_match += exact_match_score(prediction, ground_truths)
    return 100.*exact_match/len(predictions)
