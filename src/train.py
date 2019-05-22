import tensorflow as tf
from tensorflow.keras import layers
import lr_schedules
import pandas as pd
import custom_layers
import custom_losses
import custom_metrics
import preprocess
import utils


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

    batch_size = 32768

    # CLR parameters
    epochs = 10
    max_lr = 5e-3
    base_lr = max_lr / 10
    max_m = 0.98
    base_m = 0.85
    cyclical_momentum = True
    cycles = 2.35
    iterations = round(len(train_set) / batch_size * epochs)
    iterations = list(range(0, iterations + 1))
    step_size = len(iterations) / (cycles)

    clr = lr_schedules.CyclicLR(base_lr=base_lr,
                                max_lr=max_lr,
                                step_size=step_size,
                                max_m=max_m,
                                base_m=base_m,
                                cyclical_momentum=cyclical_momentum)

    callbacks = [clr]
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.0000001)

    final_activation = utils.create_rescaled_sigmoid_fn(0.0, tf.math.log(41551 * 1.20)) # min & max is from EDA notebook
    loss_fn = custom_losses.mse_log

    embedding_dim = 64
    dropout_rate = 0.1

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
    cat_features_with_vocab = preprocess.create_embedding_features(cat_vars_with_vocab, dim=embedding_dim)
    cat_features_hash_buckets = preprocess.create_embeddings_with_hash_buckets(cat_vars_hash_buckets, dim=embedding_dim)

    feature_layer = layers.DenseFeatures(cont_features + cat_features_with_vocab + cat_features_hash_buckets)

    # for viewing the feature layer
    x, y = next(iter(train_ds.take(1)))
    model = tf.keras.Sequential([
        feature_layer,
        custom_layers.DenseBlock(1024, dropout=dropout_rate),
        custom_layers.DenseBlock(512, dropout=dropout_rate),
        custom_layers.DenseBlock(256, dropout=dropout_rate),
        layers.Dense(1, activation=final_activation)
    ])
    # feature_layer(x)

    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=[custom_metrics.rmspe])

    x = next(iter(train_ds))
    model(x[0]) - x[1]
    model.summary()
    model.fit(train_ds,
              validation_data=valid_ds,
              epochs=epochs,
              callbacks=callbacks)


if __name__ == '__main__':
    main()
