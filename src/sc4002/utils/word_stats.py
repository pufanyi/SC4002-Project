from typing import Dict, Iterable

import nltk
from huggingface_hub import hf_hub_download
from nltk.tokenize import word_tokenize

from sc4002.models import Glove, tokenizer


class WordStats(object):
    def __init__(
        self,
        repo_id: str = "kcz358/glove",
        tokenizer_path: str = "glove.840B.300d/glove.840B.300d.tokenizer.json",
        checkpoint_path: str = "glove.840B.300d/glove.840B.300d.safetensors",
    ):
        self.tokenizer_path = hf_hub_download(repo_id=repo_id, filename=tokenizer_path)
        self.checkpoint_path = hf_hub_download(repo_id=repo_id, filename=checkpoint_path)
        self.glove = Glove(ckpt_path=self.checkpoint_path, tokenizer_path=self.tokenizer_path)

    def word_count(self, data: str | Iterable[str]) -> Dict[str, int]:
        total = 0
        known = 0
        unknown = 0
        if isinstance(data, str):
            data = [data]
        for text in data:
            words = word_tokenize(text)
            for word in words:
                total += 1
                if self.glove.known_word(word):
                    known += 1
                else:
                    unknown += 1
        return dict(total=total, known=known, unknown=unknown)

    def __call__(self, data: str | Iterable[str]) -> Dict[str, int]:
        return self.word_count(data)
