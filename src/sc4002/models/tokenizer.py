import json
from typing import List, Union

import torch


class Tokenizer:
    def __init__(
        self,
        tokenizer_path: str = "glove.840B.300d/glove.840B.300d.tokenizer.json",
        pad_side: str = "right",
    ) -> None:
        with open(tokenizer_path, "r") as f:
            self.tokenizer_dict = json.load(f)
        self.ids_to_tokens = {v: k for k, v in self.tokenizer_dict.items()}
        self.unk_token = "<|UNK|>"
        self.unk_id = len(self.tokenizer_dict)
        self.pad_side = pad_side
        self.pad_token = "<pad>"
        self.pad_id = len(self.tokenizer_dict) + 1
        self.ids_to_tokens[self.unk_id] = self.unk_token
        self.ids_to_tokens[self.pad_id] = self.pad_token

    def known_word(self, word: str):
        return word.lower() in self.tokenizer_dict

    @property
    def vocab_size(self):
        # +1 UNK token
        return len(self.tokenizer_dict) + 1

    def encode(self, inputs: List[str], return_tensor: str = "pt"):
        tokens = [self.greedy_match(s.lower()) for s in inputs]
        max_len = max(len(t) for t in tokens)
        tokens = [t + [self.pad_id] * (max_len - len(t)) for t in tokens]

        if return_tensor == "pt":
            return torch.tensor(tokens, dtype=torch.int32)
        return tokens

    def greedy_match(self, s: str):
        tokens = []
        i = 0
        while i < len(s):
            longest_match = None
            for j in range(i + 1, len(s) + 1):
                substring = s[i:j]
                if substring.strip() in self.tokenizer_dict:
                    if longest_match is None or len(substring) > len(longest_match):
                        longest_match = substring

            if longest_match:
                tokens.append(self.tokenizer_dict[longest_match.strip()])
                i += len(longest_match)
            else:
                i += 1
                tokens.append(len(self.tokenizer_dict))

        return tokens

    def decode(self, input_ids: Union[List[List[int]], torch.Tensor]):
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.cpu().detach().tolist()
        decode_strings = []
        for input_id in input_ids:
            decode_string = ""
            for idx in input_id:
                if idx != self.pad_id and idx != self.unk_id:
                    decode_string += self.ids_to_tokens[idx] + " "
            decode_strings.append(decode_string.strip())
        return decode_strings

    def demo(self, input: str) -> str:
        input_ids = self.encode([input])
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.cpu().detach().tolist()
        input_id = input_ids[0]
        decode_list = []
        for idx in input_id:
            decode_list.append(self.ids_to_tokens[idx])
        return " | ".join(decode_list)
