import tensorflow as tf
from tensorflow.keras import layers
import lr_schedules
import pandas as pd
import custom_layers
import custom_losses
import custom_metrics
import modelling
import utils





def main(args=None):
    train_set = pd.read_parquet('./data/train.parquet')
    valid_set = pd.read_parquet('./data/valid.parquet')

    batch_size = 32768

    # include the epoch in the file name. (uses `str.format`)
    checkpoint_path = "logdir/cp-{epoch:04d}-{val_loss:.4f}.ckpt"

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     verbose=1,
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     mode='min',
                                                     monitor='val_loss')

    # CLR parameters
    epochs = 20000
    max_lr = 5e-2
    base_lr = max_lr / 100
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

    callbacks = [cp_callback]
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.0000001)
    optimizer = 'adam'

    final_activation = utils.create_rescaled_sigmoid_fn(0.0,
                                                        tf.math.log(41551 * 1.20))  # min & max is from EDA notebook
    loss_fn = custom_losses.mse_log

    embedding_dim = 64
    dropout_rate = 0.1

    train_ds = utils.df_to_dataset(train_set, batch_size=batch_size, shuffle=True)
    valid_ds = utils.df_to_dataset(valid_set, batch_size=batch_size, shuffle=False)

    # for viewing the feature layer
    x, y = next(iter(train_ds.take(1)))

    # feature_layer(x)
    model = modelling.create_model(embedding_dim=embedding_dim,
                                   dropout_rate=dropout_rate,
                                   train_file='./data/train.parquet',
                                   final_activation=final_activation
                                   )

    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=[custom_metrics.rmspe], )

    x = next(iter(train_ds))
    model(x[0]) - x[1]
    model.summary()
    model.fit(train_ds,
              validation_data=valid_ds,
              epochs=epochs,
              callbacks=callbacks)

3
if __name__ == '__main__':
    main()
