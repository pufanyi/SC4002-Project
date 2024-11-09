import unittest

from huggingface_hub import hf_hub_download

from sc4002.models import RNN


class TestGlove(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tokenizer_path = hf_hub_download(
            repo_id="kcz358/glove", filename="glove.840B.300d/glove.840B.300d.tokenizer.json"
        )
        checkpoint_path = hf_hub_download(
            repo_id="kcz358/glove", filename="glove.840B.300d/glove.840B.300d.safetensors"
        )
        cls.rnn = RNN(ckpt_path=checkpoint_path, tokenizer_path=tokenizer_path)

    def testForward(self):
        inputs = ["Have a nice day !", "Say hello to the world"]
        outputs = self.rnn(inputs)
        self.assertAlmostEqual(float(outputs.sum()), 2)


if __name__ == "__main__":
    unittest.main()
