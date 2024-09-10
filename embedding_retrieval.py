from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, disable_caching
import numpy as np
import argparse
import datasets
import logging
import torch
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--datasets_path", type=str, help="name of the datasets path", required=True)
parser.add_argument("--output", type=str, help="name of the output directory", required=True)
parser.add_argument("--idGPU", type=int, help="ID of GPU", required=True)
args = parser.parse_args()

datasets_path = args.datasets_path
output = args.output
idGPU = args.idGPU

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('embedding_retrieval_'+os.path.basename(datasets_path)+'.log')
handler.setFormatter(logging.Formatter("%(asctime)s; %(levelname)s; %(message)s"))
logger.addHandler(handler)

logger.info("datasets path : "+datasets_path+" ; output : "+output)


# function that takes as input a batch of sentences pre-tokenized in "words" and the target word, and returns as output the corresponding word embeddings.
def recup_emb(batch,words) :

    encoded_batch = tokenizer(batch, padding=True, truncation=True, is_split_into_words=True, return_tensors="pt").to(device)

    with torch.no_grad(): 
        output = model(encoded_batch.input_ids, attention_mask=encoded_batch.attention_mask)
    
    encoded_batch = encoded_batch.to('cpu')
    last_hidden_state = output.last_hidden_state.to('cpu')

    results_emb = []
    results_words_emb = []

    for b in range(len(batch)) : 

        ids = set([i for i in encoded_batch.word_ids(b) if type(i)==int])
        meanEmbedding = []

        for i in ids : 

            if batch[b][i] in words[b] : 
                beg_t, end_t = encoded_batch.word_to_tokens(b,i)[0],encoded_batch.word_to_tokens(b,i)[1]
                embeddings_word = last_hidden_state[b][beg_t:end_t]
                meanEmbedding.append(embeddings_word.mean(dim=0).tolist())
        
        results_emb.append(meanEmbedding)

    return {"meanEmbedding":results_emb}

logger.info("embedding retrieval.")

datasets.disable_caching()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger.info("Is CUDA available:"+str(torch.cuda.is_available()))

torch.cuda.set_device(idGPU)
logger.info("idGPU : "+str(torch.cuda.current_device()))

tokenizer = AutoTokenizer.from_pretrained("Yanzhu/bertweetfr-base", cache_dir="/scratch/ltarrade/.cache/huggingface/")
model = AutoModel.from_pretrained("Yanzhu/bertweetfr-base", cache_dir="/scratch/ltarrade/.cache/huggingface/")

model.to(device)

files = sorted(glob.glob(datasets_path+"*.json"))

for nb,dataset_file in enumerate(files) : 

    fileName = os.path.basename(dataset_file).split(".")[0]
    logger.info("file "+fileName+" ("+str(nb+1)+"/"+str(len(files))+")")

    logger.info("loading dataset")
    dataset = load_dataset("json", data_files=dataset_file, cache_dir="/scratch/ltarrade/.cache/huggingface/datasets/")
    logger.info("loading dataset - ended")

    logger.info("retrieving embedding")
    dataset_withEmb = dataset.map(lambda x: recup_emb(x["words"], x["words_emb"]), batched=True, batch_size=20)
    logger.info("retrieving embedding - ended")

    justEmb = np.array([np.array(e, dtype=np.float16) for e in dataset_withEmb["train"]["meanEmbedding"]])
    np.savez_compressed(output+fileName+"_emb", justEmb)

    logger.info("file "+fileName+" ("+str(nb+1)+"/"+str(len(files))+") - ended.")

logger.info("embedding retrieval - ended.")
