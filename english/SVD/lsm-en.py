import os
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer


def make_table_and_dict(corpus_path, min_df, token_pattern=None, use_idf=True):
    with open(corpus_path, 'r', encoding='utf-8') as corpus_file:
        if token_pattern:
            vectorizer = TfidfVectorizer(analyzer='word', min_df=min_df, token_pattern=token_pattern, use_idf=use_idf)
        else:
            vectorizer = TfidfVectorizer(analyzer='word', min_df=min_df)
        data_vectorized = vectorizer.fit_transform(corpus_file)
    return data_vectorized, vectorizer.get_feature_names_out(), vectorizer.idf_


def create_table(data_vectorized, k, name):
    u, sigma, vt = svds(data_vectorized, k)

    if sigma[0] < sigma[-1]:
        u = u[:, ::-1]
        sigma = sigma[::-1]
        vt = vt[::-1, :]

    feat_dictionary = pd.DataFrame(data=np.dot(np.diag(sigma), vt).T, index=roma_dictionary)
    feat_dictionary.to_csv(os.sep.join(["dictionaries-en", f"{name}_{k}.csv"]))

    np.save(os.sep.join(["mats-en", f'data_vectorized_{k}.npy']), data_vectorized.toarray())
    np.save(os.sep.join(["mats-en", f'U_{k}.npy']), u)
    np.save(os.sep.join(["mats-en", f'sigma_{k}.npy']), np.diag(sigma))
    np.save(os.sep.join(["mats-en", f'V_transposed_{k}.npy']), vt)


def create_tables(data_vectorized, ks, name):
    for k in ks:
        create_table(data_vectorized, k, name)


regex_en_alphabet = r'([ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz]+)'
en_data_vectorized, en_dictionary, idfs = make_table_and_dict(corpus_path='../english_corpus.txt',
                                                                  min_df=3,
                                                                  token_pattern=regex_en_alphabet)

create_table(en_data_vectorized, k=1000, name="english_dictionary")
