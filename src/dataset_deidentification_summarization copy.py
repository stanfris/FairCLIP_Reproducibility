# https://huggingface.co/m42-health/med42-70b
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
import pandas as pd
import openai
import argparse
import os
import numpy as np

CACHE_DIR = 'data'

def preprocess(dataset, mode='summarize', dataset_path="./data/Harvard-FairVLMed"):
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained('DISLab/SummLlama3.2-3B', cache_dir=CACHE_DIR)
    model = transformers.LlamaForCausalLM.from_pretrained('DISLab/SummLlama3.2-3B', cache_dir=CACHE_DIR, device_map='auto') 
    prompt = 'Summarize the key details, including the presence of glaucoma, from the clinical note within 180 characters.\nClinical Note:\n'
    orig_note = pd.read_csv(os.path.join(dataset_path, "data_summary.csv")).loc[0, 'note']
    note = prompt + orig_note + '\nSummary:\n'
    print('INPUT NOTE:')
    print(note)
    batch = tokenizer(note, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        generated = model.generate(inputs = batch["input_ids"].cuda(), max_new_tokens=512, do_sample=True, top_k=50)
    processed_note = tokenizer.decode(generated[0])
    processed_note = processed_note[processed_note.find(note) + len(note):].strip('</s>')
    print('PROCESSED NOTE:')
    print(processed_note)

if __name__ == "__main__":
    dataset_path = "./data/Harvard-FairVLMed"
    dataset = sorted(os.listdir(dataset_path))
    preprocess(dataset, dataset_path)
