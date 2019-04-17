import os
import pickle
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.utils.model_zoo as model_zoo
from pytorch_pretrained_bert import BertTokenizer, BertModel
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel
from IPython import embed

import ELMo.elmo_models as elmo_models

import os

model_urls = {
    'wmt-lstm' : 'https://s3.amazonaws.com/research.metamind.io/cove/wmtlstm-b142a7f2.pth'
}

model_cache = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.torch')

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
data_path = "./data"


class Embedder: # OpenAIGPTModel
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
        with open(os.path.join(data_path, "word2idx-vectors.pkl"), "rb") as f:
            word2idx, vectors = pickle.load(f)

        self.tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        self.model = OpenAIGPTModel.from_pretrained('openai-gpt')
        self.model.eval()
        self.model.to(device)



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
        batch_size = len(sentences)
        output_len = min(max(map(len, sentences)), max_sent_len)
        embed()

        return output


class origin_good_Elmo_Embedder: # ELMO
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

        model_name = "ELMo/elmo_model_adap_big.tar"
        # TODO
        model = elmo_models.elmo_model(input_size=512,
                                       hidden_size=512,
                                       drop_p=0.,
                                       out_of_words=80000
                                       )
        ckpt = torch.load(model_name)
        print(model_name)
        model.load_state_dict(ckpt['model'])
        model.to(device)
        model.eval()

        self.elmo = model
        self.max_sent_len = 64
        self.max_word_len = 16

        self.char_pad = 256
        self.char_sos = 257
        self.char_eos = 258
        self.char_unk = 259

        self.sos = np.array([self.char_sos]+[self.char_pad]*(self.max_word_len-1))
        self.eos = np.array([self.char_eos]+[self.char_pad]*(self.max_word_len-1))
        self.word_pad = np.array([self.char_pad]*self.max_word_len)

    def _pad_and_cut(self, seq, max_leng, pad, tensor):
        if len(seq) < max_leng:
            seq = seq + [pad]*(max_leng-len(seq))
        else:
            seq = seq[:max_leng]
        if tensor:
            return np.array(seq)
        return seq

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
        batch_size = len(sentences)
        output_len = min(max(map(len, sentences)), max_sent_len)
        input_data = []

        for sent in sentences:
            out = [ self._pad_and_cut(seq=[min(ord(c), 259) for c in word],
                                      max_leng=self.max_word_len,
                                      pad=self.char_pad,
                                      tensor=False
                                      ) for word in sent]
            pad_out = self._pad_and_cut(seq=out,
                                        max_leng=output_len,
                                        pad=self.word_pad,
                                        tensor=False
                                        )

            input_data.append([self.sos]+pad_out+[self.eos])



        input_data = torch.tensor(input_data, dtype=torch.int64)
        input_data = input_data.to(device)
        embedding = self.elmo.get_contexulize_embedding(input_data)
        embedding = [item.detach().cpu().numpy().reshape(batch_size, output_len, 1, -1) for item in embedding]
        encoded_layers = np.concatenate(embedding, axis=2)
        return encoded_layers
        # return np.empty( (len(sentences), min(max(map(len, sentences)), max_sent_len), 0), dtype=np.float32)


class fake_Embedder: # handout
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
        batch_size = len(sentences)
        output_len = min(max(map(len, sentences)), max_sent_len)
        output = [self.get_embedding(senten, output_len) for senten in sentences]
        output = np.array(output)
        output = output.reshape(batch_size, output_len, 1, -1)
        return output
        # return np.empty( (len(sentences), min(max(map(len, sentences)), max_sent_len), 0), dtype=np.float32)
