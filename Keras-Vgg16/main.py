
# Data Plotting
import matplotlib.pyplot as plt


# Deep learning
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def main():

    trainer_data = ImageDataGenerator()
    train_data = trainer_data.flow_from_directory(directory="train", target_size=(224, 224))
    validator_data = ImageDataGenerator()
    validation_data = validator_data.flow_from_directory(directory="validation", target_size=(224, 224))

    '''
    agg16 neural network architecture:
    1.Convolution using 64 filters
    2.Convolution using 64 filters + Max pooling
    3.Convolution using 128 filters
    4. Convolution using 128 filters + Max pooling
    5. Convolution using 256 filters
    6. Convolution using 256 filters
    7. Convolution using 256 filters + Max pooling
    8. Convolution using 512 filters
    9. Convolution using 512 filters
    10. Convolution using 512 filters+Max pooling
    11. Convolution using 512 filters
    12. Convolution using 512 filters
    13. Convolution using 512 filters+Max pooling
    14. Fully connected with 4096 nodes
    15. Fully connected with 4096 nodes
    16. Output layer with Softmax activation with required nodes (originally a 1000).
    '''

    model = keras.Sequential()
    # 1
    model.add(
        layers.Conv2D(filters=64, kernel_size=[3, 3], activation='relu', padding='same', input_shape=(224, 224, 3)))
    # 2
    model.add(layers.Conv2D(filters=64, kernel_size=[3, 3], activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    # 3
    model.add(layers.Conv2D(filters=128, kernel_size=[3, 3], activation='relu', padding='same'))
    # 4
    model.add(layers.Conv2D(filters=128, kernel_size=[3, 3], activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    # 5
    model.add(layers.Conv2D(filters=256, kernel_size=[3, 3], activation='relu', padding='same'))
    # 6
    model.add(layers.Conv2D(filters=256, kernel_size=[3, 3], activation='relu', padding='same'))
    # 7
    model.add(layers.Conv2D(filters=256, kernel_size=[3, 3], activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    # 8
    model.add(layers.Conv2D(filters=512, kernel_size=[3, 3], activation='relu', padding='same'))
    # 9
    model.add(layers.Conv2D(filters=512, kernel_size=[3, 3], activation='relu', padding='same'))
    # 10
    model.add(layers.Conv2D(filters=512, kernel_size=[3, 3], activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    # 11
    model.add(layers.Conv2D(filters=512, kernel_size=[3, 3], activation='relu', padding='same'))
    # 12
    model.add(layers.Conv2D(filters=512, kernel_size=[3, 3], activation='relu', padding='same'))
    # 13
    model.add(layers.Conv2D(filters=512, kernel_size=[3, 3], activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    # 14
    model.add(layers.Flatten())
    model.add(layers.Dense(units=4096, activation='relu'))
    model.add(layers.Dense(units=4096, activation='relu'))
    model.add(layers.Dense(units=2, activation='softmax'))

    model.build()
    model.summary()

    learning_rate = 0.001
    model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss=keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )

    #  The model will only be saved to disk if the validation accuracy of the model in current epoch is greater than
    #  what it was in the last epoch.
    checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_accuracy', verbose=2, save_best_only=True,
                                 save_weights_only=False, mode='auto', save_freq=1)
    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

    hist = model.fit(x=train_data, steps_per_epoch=len(train_data), validation_data=validation_data,
                     validation_steps=len(validation_data), epochs=5, verbose=2, callbacks=[checkpoint, early])

    plt.plot(hist.history["acc"])
    plt.plot(hist.history['val_acc'])
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
    plt.show()


if __name__ == '__main__':
    main()
