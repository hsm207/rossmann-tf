import tensorflow as tf
from typing import List, Dict


def create_embedding_features(columns: Dict[str, List[str]], dim: int = 32):
    # he_init = tf.keras.initializers.VarianceScaling(scale=2.,
    #                                                 mode='fan_in',
    #                                                 distribution='truncated_normal')

    cat_columns = [tf.feature_column.categorical_column_with_vocabulary_list(key=col, vocabulary_list=vocab_list)
                   for col, vocab_list in columns.items()
                   ]

    embed_features = [tf.feature_column.embedding_column(cat_column, dimension=dim)
                      for cat_column in cat_columns]

    return embed_features
