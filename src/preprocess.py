import tensorflow as tf
from typing import List, Dict


def create_embedding_features(columns: Dict[str, List[str]], dim: int = 32):
    cat_columns = [tf.feature_column.categorical_column_with_vocabulary_list(key=col, vocabulary_list=vocab_list)
                   for col, vocab_list in columns.items()
                   ]

    embed_features = [tf.feature_column.embedding_column(cat_column, dimension=dim)
                      for cat_column in cat_columns]

    return embed_features


def create_embeddings_with_hash_buckets(columns: Dict[str, List[int]], dim: int = 32):
    cat_columns = [tf.feature_column.categorical_column_with_hash_bucket(key=col,
                                                                         hash_bucket_size=unique_vals,
                                                                         dtype=tf.dtypes.int32)
                   for col, unique_vals in columns.items()
                   ]

    embed_features = [tf.feature_column.embedding_column(cat_column, dimension=dim)
                      for cat_column in cat_columns]

    return embed_features
