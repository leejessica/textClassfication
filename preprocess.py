import tensorflow as tf
import random
from tqdm import tqdm
import ujson as json
from collections import Counter
import numpy as np
from codecs import open
import jieba
import logging

logging.basicconfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def word_tokenize(sent):
    doc = jieba.lcut(sent)
    return [token for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename, data_type, word_counter, char_counter, seq=';'):
    logger.info("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        lines = fh.readlines()
        for line in lines:
            total += 1
            text_tokens = word_tokenize(line.split(seq)[3])
            for token in text_tokens:
                word_counter[token] += 1
                for char in token:
                    char_counter[char] += 1
            example = {'id': total, 'text': text_tokens, 'label': line.split(seq)[1]}
            examples.append(example)
            eval_examples[str(total)] = {'text': line.split(seq)[3], 'label': line.split(seq)[1]}
        random.shuffle(examples)
        logger.info("{} text in total".format(len(examples)))
    return examples, eval_examples


def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.1) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx,
                                     token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def build_features(config, examples, data_type, out_file, word2idx_dict, char2idx_dict):
    logger.info("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    for example in tqdm(examples):
        context_idxs = np.zeros([config.max_sequence_length], dtype=np.int32)
        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        if config.use_word:
            for i, token in enumerate(example['text']):
                context_idxs[i] = _get_word(token)

        if config.use_char:
            for i, char in enumerate(example[])

        label = example['label']

        record = tf.train.Example(features=tf.train.Features(feature={
            "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            "text_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(record.SerializeToString())
    return 1


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def prepro(config):
    word_counter, char_counter = Counter()
    train_examples, train_eval = process_file(
        config.train_file, "train", word_counter, char_counter)
    dev_examples, dev_eval = process_file(
        config.dev_file, "dev", word_counter, char_counter)
    test_examples, test_eval = process_file(
        config.test_file, "test", word_counter, char_counter)

    word_emb_file = config.fasttext_file if config.fasttext else config.glove_word_file
    char_emb_file = config.glove_char_file if config.pretrained_char else None
    char_emb_size = config.glove_char_size if config.pretrained_char else None
    char_emb_dim = config.glove_dim if config.pretrained_char else config.char_dim

    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, "word", emb_file=word_emb_file, size=config.glove_word_size, vec_size=config.glove_dim)
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, "char", emb_file=char_emb_file, size=char_emb_size, vec_size=char_emb_dim)

    build_features(config, train_examples, "train",
                   config.train_record_file, word2idx_dict, char2idx_dict)
    build_features(config, dev_examples, "dev",
                   config.dev_record_file, word2idx_dict, char2idx_dict)
    build_features(config, test_examples, "test",
                   config.test_record_file, word2idx_dict, char2idx_dict, is_test=True)

    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.char_emb_file, char_emb_mat, message="char embedding")
    save(config.train_eval_file, train_eval, message="train eval")
    save(config.dev_eval_file, dev_eval, message="dev eval")
    save(config.test_eval_file, test_eval, message="test eval")
    save(config.word_dictionary, word2idx_dict, message="word dictionary")
    save(config.char_dictionary, char2idx_dict, message="char dictionary")
