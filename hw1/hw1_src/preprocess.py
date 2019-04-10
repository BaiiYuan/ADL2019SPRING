import torch
import re
import io
import os
import json
import gensim
import pickle
from gensim.models import Word2Vec
from IPython import embed
from sys import stdout
import nltk

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

data_path = "./data"

all_sentence = []
all_words = []

def load_vectors(fname, word_set):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    cou = 2 # <pad> as 0
    word2idx = {'<pad>': 0, '<unk>': 1}
    vectors = []
    for line in fin:
        tokens = line.rstrip().split(' ')
        if tokens[0] in word_set:
            word2idx[tokens[0]] = cou
            cou += 1
            stdout.write("\r{}".format(cou))
            vectors.append([float(v) for v in tokens[1:]])
    print(" ")
    vectors = torch.tensor(vectors)
    vectors = torch.cat([torch.nn.init.uniform_(torch.empty(1, 300)), vectors], dim=0)
    vectors = torch.cat([torch.zeros(1, 300), vectors], dim=0)

    return word2idx, vectors

def load_vectors_gensim(fname, word_set):
    model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
    all_word = list(model.index2word)
    cou = 2 # <pad> as 0
    word2idx = {'<pad>': 0, '<unk>': 1}
    vectors = []
    for line in all_word:
        if line in word_set:
            word2idx[line] = cou
            cou += 1
            stdout.write("\r{}".format(cou))
            vectors.append(model.get_vector(line))
    print(" ")
    vectors = torch.tensor(vectors)
    vectors = torch.cat([torch.nn.init.uniform_(torch.empty(1, 300)), vectors], dim=0)
    vectors = torch.cat([torch.zeros(1, 300), vectors], dim=0)

    return word2idx, vectors

def preprocessSentence(raw_sentence):
    a = raw_sentence
    # a = re.sub(r"([!?])", r" \1 ", raw_sentence)
    a = re.sub(r"[^a-zA-Z!?',.]+", r" ", a)
    # a = re.sub(r"\s+", r" ", a).strip()
    a = a.lower()
    a = nltk.word_tokenize(a)
    # print(raw_sentence);print(a);print("----")
    a.append("<pad>")
    return a

def preprocessData(raw_datas):
    ret_data = []
    cou = 0
    for data in raw_datas:
        stdout.write("\r{}".format(cou))
        cou+=1
        tmp = {}
        ans = data['options-for-correct-answers'][0]
        ansID = ans['candidate-id']
        correct_answer = preprocessSentence(ans['utterance'])
        records = data['messages-so-far']
        options = data['options-for-next']
        thisID = data['example-id']

        wrong_answer = []
        for i, option in enumerate(options):
            if option['candidate-id'] == ansID:
                pass
                # print(option['utterance']);print(correct_answer);print("-----")
            else:
                wrong_answer.append(preprocessSentence(option['utterance']))

        tmp['thisID'] = thisID
        tmp['records'] = [preprocessSentence(record['utterance']) for record in records]
        tmp['wrong_answer'] = wrong_answer
        tmp['ansID'] = ansID
        tmp['correct_answer'] = correct_answer

        ret_data.append(tmp)
        all_sentence.extend([i for i in tmp['records']])
        all_sentence.extend([i for i in [tmp['correct_answer']]])
        all_sentence.extend([i for i in tmp['wrong_answer']])
        assert(len(tmp['wrong_answer'])==99)
    print(" ")
    return ret_data

def preprocessTestData(raw_datas):
    ret_data = []
    cou = 0
    for data in raw_datas:
        stdout.write("\r{}".format(cou))
        cou+=1
        tmp = {}
        records = data['messages-so-far']
        options = data['options-for-next']
        thisID = data['example-id']

        wrong_answer = []
        for i, option in enumerate(options):
            wrong_answer.append(preprocessSentence(option['utterance']))

        tmp['thisID'] = thisID
        tmp['records'] = [preprocessSentence(record['utterance']) for record in records]
        tmp['wrong_answer'] = wrong_answer

        ret_data.append(tmp)
        all_sentence.extend([i for i in tmp['records']])
        all_sentence.extend([i for i in tmp['wrong_answer']])
        assert(len(tmp['wrong_answer'])==100)
    print(" ")
    return ret_data

def morePreprocessData(processed_datas, all_vocab, word2idx):
    ret_data = []
    cou = 0
    for data in processed_datas:
        stdout.write("\r{}".format(cou))
        cou+=1
        records = data['records']
        data['records'] = [ [word2idx.get(i, 1) for i in record] for record in records]
        correct_answer = data['correct_answer']
        data['correct_answer'] = [ [word2idx.get(i, 1) for i in record] for record in [correct_answer]][0]
        wrong_answer = data['wrong_answer']
        data['wrong_answer'] = [ [word2idx.get(i, 1) for i in record] for record in wrong_answer]
        ret_data.append(data)
    print(" ")
    return ret_data

def morePreprocessTestData(processed_datas, all_vocab, word2idx):
    ret_data = []
    cou = 0
    for data in processed_datas:
        stdout.write("\r{}".format(cou))
        cou+=1
        records = data['records']
        data['records'] = [ [word2idx.get(i, 1) for i in record] for record in records]
        wrong_answer = data['wrong_answer']
        data['wrong_answer'] = [ [word2idx.get(i, 1) for i in record] for record in wrong_answer]
        ret_data.append(data)
    print(" ")
    return ret_data

def main():
    print("loading data ...")
    with open(os.path.join(data_path, "train.json")) as f:
        train_raw = json.load(f)
    with open(os.path.join(data_path, "valid.json")) as f:
        valid_raw = json.load(f)
    with open(os.path.join(data_path, "test.json")) as f:
        test_raw = json.load(f)

    print("processing data ...")
    train_data = preprocessData(train_raw)
    valid_data = preprocessData(valid_raw)
    test_data = preprocessTestData(test_raw)

    word_set = set([])
    for sentence in all_sentence:
        word_set |= set(sentence)


    print(len(list(word_set)))
    print("loading pre-trained model ...")
    # word2idx, vectors = load_vectors("./crawl-300d-2M.vec", word_set)
    word2idx, vectors = load_vectors_gensim("./GoogleNews-vectors-negative300.bin", word_set)

    with open(os.path.join(data_path, "dict&vectors.pkl"), "wb") as f:
        pickle.dump([word2idx, vectors], f)

    ## Gensim ##
    # print("building word model ...")
    # word_model = Word2Vec(all_sentence, size=1000, window=5, min_count=5, workers=16)
    # word_model.save("Word2Vec_V2.h5")
    # word_model = Word2Vec.load("Word2Vec_V2.h5")
    # vectors = word_model.wv
    ## Gensim ##

    print("more processing data ... (Clean the OOV)")
    train_data = morePreprocessData(train_data, word_set, word2idx)
    valid_data = morePreprocessData(valid_data, word_set, word2idx)
    test_data = morePreprocessTestData(test_data, word_set, word2idx)


    with open(os.path.join(data_path, "train.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    with open(os.path.join(data_path, "valid.pkl"), "wb") as f:
        pickle.dump(valid_data, f)
    with open(os.path.join(data_path, "test.pkl"), "wb") as f:
        pickle.dump(test_data, f)

if __name__ == '__main__':
    main()