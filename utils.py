import numpy as np
import torch
from gensim.utils import simple_preprocess
from sklearn.datasets import fetch_20newsgroups
from typing import Dict, List, Tuple, Any

from torch import Tensor

MAX_WORDS = 2000
GLOVE_PATH = "glove.6B.100d.txt"


def load_20ng() -> Tuple[List[Any], Any]:
    # getting 20 newsgroup dataset
    categories = ["alt.atheism", "sci.med", "sci.electronics", "comp.graphics", "talk.politics.guns", "sci.crypt"]
    ng_text, ng_class = fetch_20newsgroups(data_home="20news-bydate", categories=categories, return_X_y=True)
    subset_ind = np.array([i for i, doc in enumerate(ng_text) if len(doc) > 0 and len(doc.split(" ", 5)) > 5])
    return [doc[:MAX_WORDS] for i, doc in enumerate(ng_text) if i in subset_ind], ng_class[subset_ind]


def load_glove(ng_text: list) -> Tuple[Tensor, dict]:
    ng_tags = set()
    for doc in ng_text:
        ng_tags.update(set(simple_preprocess(doc)))
    glove_tags = set([l.split(" ", 1)[0] for l in open(GLOVE_PATH).read().splitlines()])
    glove_tags &= ng_tags

    # collecting glove vectors
    glove_vectors = {}
    for line in open(GLOVE_PATH).read().splitlines():
        tokens = line.split(" ")
        if tokens[0] in glove_tags:
            glove_vectors[tokens[0]] = torch.tensor([float(i) for i in tokens[1:]], dtype=torch.float)
    word_to_idx = {word: idx for idx, word in enumerate(glove_vectors)}

    # embedding matrix
    embedding_dim = len(list(glove_vectors.values())[0])  # Accessing the vector size of each dimension
    vocab_size = len(glove_vectors)

    embedding_matrix = torch.zeros((vocab_size, embedding_dim))
    for word, idx in word_to_idx.items():
        embedding_matrix[idx] = glove_vectors[word]
    return embedding_matrix, word_to_idx


def doc2ind(doc: str, word_to_idx: Dict[str, Tensor]) -> List[Tensor]:
    """
    Convert 20NG document into glove vectors indices
    :param word_to_idx: mapping of each word to idx in Glove
    :param doc: a document in 20NG data-set
    :return:
    """
    tokens = simple_preprocess(doc)
    return [word_to_idx[token] for i, token in enumerate(tokens) if token in word_to_idx]
