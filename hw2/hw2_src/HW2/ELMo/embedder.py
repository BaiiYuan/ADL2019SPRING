import numpy as np

data_path = "../data"

class Embedder:
    """
    The class responsible for loading a pre-trained ELMo model and provide the ``embed``
    functionality for downstream BCN model.

    You can modify this class however you want, but do not alter the class name and the
    signature of the ``embed`` function. Also, ``__init__`` function should always have
    the ``ctx_emb_dim`` parameter.
    """

    def __init__(self, ctx_emb_dim):
        """
        The value of the parameters should also be specified in the BCN model config.
        """
        self.ctx_emb_dim = ctx_emb_dim
        # TODO
        with open(os.path.join(data_path, "word2idx-vectors.pkl"), "rb") as f:
            word2idx, vectors = pickle.load(f)

        self.word2idx = word2idx
        self.vectors = vectors


    def get_embedding(senten, max_sent_len):
        if len(senten) < max_sent_len:
            senten = ["<pad>"]*max_sent_len-len(senten) + senten
        else:
            senten = senten[:max_sent_len]

        return [np.array(self.vectors[self.word2idx[w]]) for w in senten]

    def __call__(self, sentences, max_sent_len):
        """
        Generate the contextualized embedding of tokens in ``sentences``.

        Parameters
        ----------
        sentences : ``List[List[str]]``
            A batch of tokenized sentences.
        max_sent_len : ``int``
            All sentences must be truncated to this length.

        Returns
        -------
        ``np.ndarray(``
            The contextualized embedding of the sentence tokens.

            The ndarray shape must be
            ``(len(sentences), min(max(map(len, sentences)), max_sent_len), self.ctx_emb_dim)``
            and dtype must be ``np.float32``.
        """
        # TODO
        output_len = min(max(map(len, sentences)), max_sent_len)
        output = [get_embedding(senten, output_len) for senten in sentences]
        return np.empty(
            (len(sentences), min(max(map(len, sentences)), max_sent_len), 0), dtype=np.float32)
