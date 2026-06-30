from tensorflow.keras import layers, models, regularizers

from .config import REGULARIZATION_L2


def build_model(input_shape=(128, 128, 3), num_classes=4):
    regularizer = regularizers.l2(REGULARIZATION_L2)

    model = models.Sequential()
    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=input_shape,
                            kernel_regularizer=regularizer, bias_regularizer=regularizer))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                            kernel_regularizer=regularizer, bias_regularizer=regularizer))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                            kernel_regularizer=regularizer, bias_regularizer=regularizer))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                            kernel_regularizer=regularizer, bias_regularizer=regularizer))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                            kernel_regularizer=regularizer, bias_regularizer=regularizer))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=128, activation='relu', kernel_regularizer=regularizer, bias_regularizer=regularizer))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(units=256, activation='relu', kernel_regularizer=regularizer, bias_regularizer=regularizer))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(units=num_classes, activation='softmax', kernel_regularizer=regularizer, bias_regularizer=regularizer))
    return model


def compile_model(model, optimizer='adam', loss='categorical_crossentropy', metrics=None):
    if metrics is None:
        metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
