import os
import pickle
import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from IPython import embed

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
data_path = "./data"

class Embedder: # BERT
    """
    The class responsible for loading a pre-trained ELMo model and provide the ``embed``
    functionality for downstream BCN model.

    You can modify this class however you want, but do not alter the class name and the
    signature of the ``embed`` function. Also, ``__init__`` function should always have
    the ``ctx_emb_dim`` parameter.
    """

    def __init__(self, n_ctx_embs, ctx_emb_dim):
        """
        The value of the parameters should also be specified in the BCN model config.
        """
        self.n_ctx_embs = n_ctx_embs
        self.ctx_emb_dim = ctx_emb_dim
        # TODO
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert = BertModel.from_pretrained('bert-base-uncased')
        bert.eval()
        bert.to(device)

        self.tokenizer = tokenizer
        self.bert = bert

    def cut_and_pad(self, indexed_token, max_sent_len):
        if len(indexed_token) < max_sent_len:
            indexed_token = indexed_token + [0]*(max_sent_len-len(indexed_token))
        else:
            indexed_token = indexed_token[:max_sent_len]
        return indexed_token

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
            ``(len(sentences), min(max(map(len, sentences)), max_sent_len), self.n_ctx_embs, self.ctx_emb_dim)``
            and dtype must be ``np.float32``.
        """

        # TODO
        output_len = min(max(map(len, sentences)), max_sent_len)

        indexed_tokens = [self.tokenizer.convert_tokens_to_ids(tokenized_text) for tokenized_text in sentences]
        # output_len = min(max(map(len, indexed_tokens)), max_sent_len)
        indexed_tokens = [self.cut_and_pad(indexed_token, output_len) for indexed_token in indexed_tokens]
        tokens_tensor = torch.tensor(indexed_tokens).to(device)
        with torch.no_grad():
            encoded_layers, _ = self.bert(tokens_tensor)

        encoded_layers = [encoded_layer.cpu().detach().numpy().reshape(-1, output_len, 1, 768) for encoded_layer in encoded_layers]
        encoded_layers = np.concatenate(encoded_layers[-1:], axis=2)
        return encoded_layers
        # return np.empty( (len(sentences), min(max(map(len, sentences)), max_sent_len), 0), dtype=np.float32)


class elmo_Embedder: # handout
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


    def get_embedding(self, senten, max_sent_len):
        if len(senten) < max_sent_len:
            senten = ["<pad>"]*(max_sent_len-len(senten)) + senten
        else:
            senten = senten[:max_sent_len]

        return [np.array(self.vectors[self.word2idx.get(w, 1)]) for w in senten]

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
            ``(len(sentences), min(max(map(len, sentences)), max_sent_len), self.n_ctx_embs, self.ctx_emb_dim)``
            and dtype must be ``np.float32``.
        """

        # TODO
        output_len = min(max(map(len, sentences)), max_sent_len)
        output = [self.get_embedding(senten, output_len) for senten in sentences]
        output = np.array(output)
        output.reshape(list(output.shape)+[1])
        return output
        # return np.empty( (len(sentences), min(max(map(len, sentences)), max_sent_len), 0), dtype=np.float32)
