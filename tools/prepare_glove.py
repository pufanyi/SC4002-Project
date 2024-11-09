import glob
import json
import os
import subprocess
import zipfile

import torch
from safetensors.torch import save_model
from tqdm import tqdm

FILES = ["glove.6B", "glove.42B.300d", "glove.840B.300d", "glove.twitter.27B"]


def download_glove_weights(file_name: str, save_path: str):
    """
    Download weights from url and save to save_path
    """
    if not os.path.exists(os.path.join(save_path, file_name)):
        os.makedirs(save_path, exist_ok=True)
        subprocess.run(
            [f"wget https://nlp.stanford.edu/data/{file_name} -O {os.path.join(save_path, file_name)}"],
            shell=True,
        )
    else:
        print(f"{file_name} already exists")
    return os.path.join(save_path, file_name)


def extract_zip_files(file_name: str, dir_name: str):
    if not os.path.exists(dir_name):
        print(f"Extracting zip files {file_name} to {dir_name}...")
        with zipfile.ZipFile(file_name, "r") as zip_ref:
            zip_ref.extractall(dir_name)
        return dir_name
    else:
        print(f"{dir_name} already exists")
        return dir_name


def prepare_weights_and_tokenizer(file_name: str):
    d = int(file_name.split(".")[-2][:-1])
    with open(file_name, "r") as f:
        lines = [line.rstrip() for line in f]

    weights = []
    tokenizer = {}
    for idx, line in enumerate(tqdm(lines, desc="Processing weights and tokens..")):
        data = line.rsplit(" ", d + 1)
        word = data[0]
        vector = data[1:]
        vector = [float(v) for v in vector]
        weights.append(vector)
        tokenizer[word] = idx
    weights = torch.tensor(weights)
    embeddings = torch.nn.Embedding.from_pretrained(weights)
    return embeddings, tokenizer


if __name__ == "__main__":
    for file in FILES:
        file_name = download_glove_weights(f"{file}.zip", save_path="./checkpoints")
        dir_name = extract_zip_files(file_name, dir_name=f"./checkpoints/{file}")
        downloaded_weights = glob.glob(f"./checkpoints/{file}/*.txt")
        for weight in downloaded_weights:
            safetensors_file_name = weight.replace(".txt", ".safetensors")
            tokenizer_file_name = weight.replace(".txt", ".tokenizer.json")
            if not os.path.exists(safetensors_file_name) or not os.path.exists(tokenizer_file_name):
                embeddings, tokenizer = prepare_weights_and_tokenizer(weight)
                save_model(embeddings, safetensors_file_name)
                with open(tokenizer_file_name, "w") as f:
                    json.dump(tokenizer, f)
            else:
                print(f"{safetensors_file_name} already exists")
                print(f"{tokenizer_file_name} already exists")
