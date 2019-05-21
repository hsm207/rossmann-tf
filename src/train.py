import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import custom_layers
import numpy as np
import custom_losses
import custom_metrics
import preprocess


def df_to_dataset(dataframe: pd.DataFrame, shuffle: bool = True, batch_size: int = 32) -> tf.data.Dataset:
    """
    Converts a pandas dataframe to a tf.dataset
    :param dataframe: Dataframe to convert to a  dataset
    :param shuffle: Whether to shuffle the dataset or not
    :param batch_size: batch size
    :return: A tf.dataset version of dataframe
    """

    df = dataframe.copy()
    df.pop('Date')
    labels = df.pop('Sales').astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))

    ds = ds.batch(batch_size)

    return ds


def main(args=None):
    train_set = pd.read_parquet('./data/train.parquet')
    valid_set = pd.read_parquet('./data/valid.parquet')

    batch_size = 8192
    train_ds = df_to_dataset(train_set, batch_size=batch_size, shuffle=True)
    valid_ds = df_to_dataset(valid_set, batch_size=batch_size, shuffle=False)

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

    cont_features = [tf.feature_column.numeric_column(var) for var in cont_vars]
    cat_features_with_vocab = preprocess.create_embedding_features(cat_vars_with_vocab, dim=32)
    feature_layer = layers.DenseFeatures(cont_features + cat_features_with_vocab)

    # for viewing the feature layer
    x, y = next(iter(train_ds.take(1)))
    model = tf.keras.Sequential([
        feature_layer,
        custom_layers.DenseBlock(1024, dropout=0.5),
        custom_layers.DenseBlock(512, dropout=0.5),
        custom_layers.DenseBlock(256, dropout=0.5),
        layers.Dense(1, activation='relu')
    ])

    model.compile(optimizer='adam',
                  loss=custom_losses.mean_squared_percentage_error,
                  metrics=[custom_metrics.rmspe])

    x = next(iter(train_ds))
    model(x[0]) - x[1]
    model.summary()
    model.fit(train_ds,
              validation_data=valid_ds,
              epochs=5)


if __name__ == '__main__':
    main()
