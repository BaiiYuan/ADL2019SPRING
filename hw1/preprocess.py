import torch
import re
import os
import json
import gensim
import pickle
from gensim.models import Word2Vec
from IPython import embed

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

data_path = "./data"

all_sentence = []

def preprocessSentence(raw_sentence):
    a = re.sub(r"([!?])", r" \1 ", raw_sentence)
    a = re.sub(r"[^a-zA-Z!?]+", r" ", a)
    a = re.sub(r"\s+", r" ", a).strip()
    a = a.lower()
    # print(raw_sentence);print(a);print("----")
    return "<s> "+ a +" <e>"

def preprocessData(raw_datas):
    ret_data = []
    for data in raw_datas:
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
        all_sentence.extend([i.split() for i in tmp['records']])
        all_sentence.extend([i.split() for i in [tmp['correct_answer']]])
        all_sentence.extend([i.split() for i in tmp['wrong_answer']])
        assert(len(tmp['wrong_answer'])==99)

    return ret_data

def preprocessTestData(raw_datas):
    ret_data = []
    for data in raw_datas:
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
        all_sentence.extend([i.split() for i in tmp['records']])
        all_sentence.extend([i.split() for i in tmp['wrong_answer']])
        assert(len(tmp['wrong_answer'])==100)

    return ret_data

def embedding_filter(arr, vectors):
    return 

def morePreprocessData(processed_datas, vectors):
    ret_data = []
    for data in processed_datas:
        records = data['records']
        data['records'] = [ [i for i in record.split() if i in vectors.vocab] for record in records]
        correct_answer = data['correct_answer']
        data['correct_answer'] = [ [i for i in record.split() if i in vectors.vocab] for record in [correct_answer]][0]
        wrong_answer = data['wrong_answer']
        data['wrong_answer'] = [ [i for i in record.split() if i in vectors.vocab] for record in wrong_answer]

        ret_data.append(data)
    return ret_data

def morePreprocessTestData(processed_datas, vectors):
    ret_data = []
    for data in processed_datas:
        records = data['records']
        data['records'] = [ [i for i in record.split() if i in vectors.vocab] for record in records]
        wrong_answer = data['wrong_answer']
        data['wrong_answer'] = [ [i for i in record.split() if i in vectors.vocab] for record in wrong_answer]

        ret_data.append(data)
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

    print("building word model ...")
    word_model = Word2Vec(all_sentence, size=1000, window=5, min_count=5, workers=16)
    word_model.save("Word2Vec_V2.h5")
    word_model = Word2Vec.load("Word2Vec_V2.h5")
    vectors = word_model.wv

    print("more processing data ...")
    train_data = morePreprocessData(train_data, vectors)
    valid_data = morePreprocessData(valid_data, vectors)
    test_data = morePreprocessTestData(test_data, vectors)

    with open(os.path.join(data_path, "train.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    with open(os.path.join(data_path, "valid.pkl"), "wb") as f:
        pickle.dump(valid_data, f)
    with open(os.path.join(data_path, "test.pkl"), "wb") as f:
        pickle.dump(test_data, f)

    embed()

if __name__ == '__main__':
    main()