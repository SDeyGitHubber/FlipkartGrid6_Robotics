import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(path, target_size=(256, 256), batch_size=32):
    data_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode="nearest",
        horizontal_flip=True
    )
    
    data = data_gen.flow_from_directory(
        path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical"
    )
    return data