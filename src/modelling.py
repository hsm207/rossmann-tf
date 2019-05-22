from tensorflow.keras import layers
import pandas as pd
import preprocess
import tensorflow as tf
import custom_layers
from typing import Union, Callable, Any


def build_feature_layer(embedding_dim: int, train_file: str):
    train_set = pd.read_parquet(train_file)

    # list of continuous variables
    cont_vars = ['AfterStateHoliday',
                 'BeforeStateHoliday',
                 'CompetitionDistance',
                 'Max_Humidity',
                 'Max_TemperatureC',
                 'Max_Wind_SpeedKm_h',
                 'Mean_Humidity',
                 'Mean_TemperatureC',
                 'Mean_Wind_SpeedKm_h',
                 'Min_Humidity',
                 'Min_TemperatureC',
                 'trend',
                 'trend_DE']

    # list of categorical variables that have a vocabulary
    cat_vars_with_vocab = {
        'Assortment': ['a' 'c' 'b'],
        'CloudCover': ['1.0' '4.0' '2.0' '6.0' '5.0' 'nan' '3.0' '8.0' '7.0' '0.0'],
        'Events': ['Fog' '-1' 'Rain' 'Rain-Thunderstorm' 'Fog-Rain' 'Rain-Hail-Thunderstorm'
                   'Fog-Rain-Thunderstorm' 'Thunderstorm' 'Rain-Hail' 'Fog-Thunderstorm'
                   'Rain-Snow' 'Fog-Rain-Hail-Thunderstorm' 'Snow' 'Rain-Snow-Hail'
                   'Rain-Snow-Hail-Thunderstorm' 'Rain-Snow-Thunderstorm' 'Fog-Rain-Snow'
                   'Fog-Snow' 'Snow-Hail' 'Fog-Rain-Snow-Hail' 'Fog-Rain-Hail'
                   'Fog-Snow-Hail'],
        'PromoInterval': ['-1' 'Jan,Apr,Jul,Oct' 'Feb,May,Aug,Nov' 'Mar,Jun,Sept,Dec'],
        'State': ['HE' 'TH' 'NW' 'BE' 'SN' 'SH' 'HB,NI' 'BY' 'BW' 'RP' 'ST' 'HH'],
        'StoreType': ['c' 'a' 'd' 'b']
    }

    # list of categorical variables that do not have a vocabulary
    cat_vars_no_vocab = [
        'CompetitionMonthsOpen',
        'CompetitionOpenSinceYear',
        'Day',
        'DayOfWeek',
        'Month',
        'Promo',
        'Promo2SinceYear',
        'Promo2Weeks',
        'Promo_bw',
        'Promo_fw',
        'SchoolHoliday',
        'SchoolHoliday_bw',
        'SchoolHoliday_fw',
        'StateHoliday',
        'StateHoliday_bw',
        'StateHoliday_fw',
        'Store',
        'Week',
        'Year',
        'CompetitionDistance_na'
    ]

    # compute the number of buckets for the categorical variables without a vocabulary
    cat_vars_hash_buckets = dict(train_set[cat_vars_no_vocab].nunique())

    # compute the mean and standard deviation of each continuous variables
    cont_vars_norm_params = pd.concat([train_set[cont_vars].mean(), train_set[cont_vars].std()], axis=1) \
        .reset_index() \
        .values \
        .tolist()

    cont_features = preprocess.create_norm_continuos_features(cont_vars_norm_params)
    cat_features_with_vocab = preprocess.create_embedding_features(cat_vars_with_vocab, dim=embedding_dim)
    cat_features_hash_buckets = preprocess.create_embeddings_with_hash_buckets(cat_vars_hash_buckets, dim=embedding_dim)

    feature_layer = layers.DenseFeatures(cont_features + cat_features_with_vocab + cat_features_hash_buckets)

    return feature_layer


def create_model(embedding_dim: int,
                 dropout_rate: float,
                 train_file: str,
                 final_activation: Union[str, Callable[[Any], Any]]):
    feature_layer = build_feature_layer(embedding_dim=embedding_dim, train_file=train_file)

    model = tf.keras.Sequential([
        feature_layer,
        custom_layers.DenseBlock(1024, dropout=dropout_rate),
        custom_layers.DenseBlock(512, dropout=dropout_rate),
        custom_layers.DenseBlock(256, dropout=dropout_rate),
        layers.Dense(1, activation=final_activation)
    ])

    return model
