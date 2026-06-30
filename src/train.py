import tensorflow as tf


def train_model(model, train_data, validation_data=None, epochs=100, model_path=None):
    callbacks = []
    if validation_data is not None and model_path is not None:
        callbacks.append(
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
        callbacks=callbacks,
    )

    if model_path is not None and not callbacks:
        model.save(model_path)

    return history
