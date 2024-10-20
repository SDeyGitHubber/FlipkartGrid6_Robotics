from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

def build_cnn_model(input_shape=(256, 256, 3), num_classes=12):
    model = Sequential()

    model.add(Conv2D(256, (3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPool2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPool2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPool2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPool2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    return model