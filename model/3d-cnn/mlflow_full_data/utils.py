import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.data import Dataset

class_mapping = {
    "canola": 0, "kochia": 1, "ragweed": 2,
    "redroot": 3, "soybean": 4, "sugarbeet": 5, "waterhemp": 6
}

def make_generator(hdf5_path, indices):
    def generator():
        with h5py.File(hdf5_path, "r") as f:
            for idx in indices:
                X = f[f"X_{idx}"][:]
                y = f[f"y_{idx}"][()]
                labels = np.full(X.shape[0], y, dtype=np.int32)
                yield X, labels
    return generator

def make_datasets(hdf5_path, batch_size=32, T = 70):
    with h5py.File(hdf5_path, 'r') as f:
        train_idx = f.attrs["train_indices"]
        val_idx = f.attrs["val_indices"]
        test_idx = f.attrs["test_indices"]

    print("Partition: ", val_idx,test_idx,train_idx)
    ds_train = Dataset.from_generator(
        make_generator(hdf5_path, train_idx),
        output_signature=(
            tf.TensorSpec(shape=(None,19,19,T), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)  #(None,7)
        )
    ).flat_map(lambda x,y: Dataset.from_tensor_slices((x,y))).repeat().shuffle(12000, reshuffle_each_iteration=True).batch(batch_size)

    ds_val = Dataset.from_generator(
        make_generator(hdf5_path, val_idx),
        output_signature=(
            tf.TensorSpec(shape=(None,19,19,T), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    ).flat_map(lambda x,y: Dataset.from_tensor_slices((x,y))).batch(batch_size)
    ds_test = Dataset.from_generator(
        make_generator(hdf5_path, test_idx),
        output_signature=(
            tf.TensorSpec(shape=(None,19,19,T), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    ).flat_map(lambda x,y: Dataset.from_tensor_slices((x,y))).batch(batch_size)

    return ds_train, ds_val, ds_test
T=70
def build_model(input_shape = (19,19,T)):
    inp = layers.Input(input_shape)
    inp = layers.Reshape((19,19,T,1))(inp)
    x = layers.Conv3D(64, (3,3,9), padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2,2,2))(x)
    x = layers.Conv3D(128, (3,3,5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(200, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(7, activation="softmax")(x)
    model = models.Model(inp, output)
    return model
