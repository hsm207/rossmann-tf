import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import custom_layers
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
    labels = df.pop('Sales')

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

    cat_vars_hash_buckets = dict(train_set[cat_vars_no_vocab].nunique())

    cont_vars_norm_params = pd.concat([train_set[cont_vars].mean(), train_set[cont_vars].std()], axis=1) \
        .reset_index() \
        .values.tolist()

    cont_features = preprocess.create_norm_continuos_features(cont_vars_norm_params)
    # cont_features = [tf.feature_column.numeric_column(var) for var in cont_vars]
    cat_features_with_vocab = preprocess.create_embedding_features(cat_vars_with_vocab, dim=32)
    cat_features_hash_buckets = preprocess.create_embeddings_with_hash_buckets(cat_vars_hash_buckets, dim=32)

    feature_layer = layers.DenseFeatures(cont_features + cat_features_with_vocab + cat_features_hash_buckets)

    # for viewing the feature layer
    x, y = next(iter(train_ds.take(1)))
    model = tf.keras.Sequential([
        feature_layer,
        custom_layers.DenseBlock(1024, dropout=0.5),
        custom_layers.DenseBlock(512, dropout=0.5),
        custom_layers.DenseBlock(256, dropout=0.5),
        layers.Dense(1, activation='relu')
    ])
    feature_layer(x)

    model.compile(optimizer='adam',
                  loss=custom_losses.mean_squared_percentage_error,
                  metrics=[custom_metrics.rmspe])

    model.summary()
    model.fit(train_ds,
              validation_data=valid_ds,
              epochs=5)


if __name__ == '__main__':
    main()
