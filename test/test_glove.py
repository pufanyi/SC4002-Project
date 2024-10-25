import unittest

from huggingface_hub import hf_hub_download

from sc4002.models import Glove, Tokenizer


class TestGlove(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tokenizer_path = hf_hub_download(repo_id="kcz358/glove", filename="glove.6B/glove.6B.300d.tokenizer.json")
        checkpoint_path = hf_hub_download(repo_id="kcz358/glove", filename="glove.6B/glove.6B.300d.safetensors")
        cls.glove = Glove(ckpt_path=checkpoint_path, tokenizer_path=tokenizer_path)

    def testTokenizer(self):
        inputs = ["Have a nice day !", "Say hello to the world"]
        tokens = self.glove.tokenizer.encode(inputs)
        words = self.glove.tokenizer.decode(tokens)
        for idx, word in enumerate(words):
            assert word.strip().lower() == inputs[idx].strip().lower()


if __name__ == "__main__":
    unittest.main()
