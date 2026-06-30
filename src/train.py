import tensorflow as tf

from .config import EARLY_STOPPING_PATIENCE, MIN_LR, REDUCE_LR_PATIENCE


def train_model(model, train_data, validation_data=None, epochs=100, model_path=None, callbacks=None):
    callback_list = list(callbacks or [])

    if validation_data is not None:
        callback_list.append(
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            )
        )
        callback_list.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=REDUCE_LR_PATIENCE,
                min_lr=MIN_LR,
                verbose=1,
            )
        )

        if model_path is not None:
            callback_list.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(model_path),
                    save_best_only=True,
                    monitor='val_loss',
                    mode='min',
                    verbose=1,
                )
            )

    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callback_list,
    )

    if model_path is not None and not callback_list:
        model.save(model_path)

    return history
