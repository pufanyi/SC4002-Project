from re import split
from typing import Dict, Iterable

from huggingface_hub import hf_hub_download

from sc4002.models import Glove


class WordStats(object):
    def __init__(self):
        tokenizer_path = hf_hub_download(
            repo_id="kcz358/glove", filename="glove.840B.300d/glove.840B.300d.tokenizer.json"
        )
        checkpoint_path = hf_hub_download(
            repo_id="kcz358/glove", filename="glove.840B.300d/glove.840B.300d.safetensors"
        )
        self.glove = Glove(ckpt_path=checkpoint_path, tokenizer_path=tokenizer_path)

    def word_count(self, data: str | Iterable[str]) -> Dict[str, int]:
        total = 0
        known = 0
        unknown = 0
        if isinstance(data, str):
            data = [data]
        for s in data:
            for word in split(r"\b\w+\b", s):
                print(word)
                total += 1
                if self.glove.known_word(word):
                    known += 1
                else:
                    unknown += 1
        return dict(total=total, known=known, unknown=unknown)
