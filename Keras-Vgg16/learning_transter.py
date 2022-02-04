
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

    vgg16_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)

    vgg16_model.build(input_shape=(224, 224, 3))
    print(vgg16_model.summary())

    model = keras.Sequential()
    for layer in vgg16_model.layers[:-1]:
        model.add(layer)

    for layer in model.layers:
        layer.trainable = False

    model.add(layers.Dense(units=2, activation='softmax'))

    print(model.summary())

    learning_rate = 0.001
    model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss=keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )

    #  The model will only be saved to disk if the validation accuracy of the model in current epoch is greater than
    #  what it was in the last epoch.
    checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_accuracy', verbose=2, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=2, mode='auto')

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
