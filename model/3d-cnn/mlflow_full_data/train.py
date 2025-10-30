import argparse
import tensorflow as tf
from tensorflow.keras import callbacks, losses, optimizers
from tensorflow.keras.models import load_model
import mlflow
import mlflow.tensorflow
from utils import build_model, make_datasets
import h5py
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pca_channels", type=int, default=70)
    parser.add_argument("--model_path", type = str, default=None)
    args = parser.parse_args()


    with h5py.File(args.hdf5_path, 'r') as f:
        total_samples = sum(f[f'X_{i}'].shape[0] for i in range(len(f) // 2))
        train_idx = f.attrs["train_indices"]
        val_idx = f.attrs["val_indices"]
        test_idx = f.attrs["test_indices"]
        all_images = sum([len(x) for x in [train_idx,val_idx,test_idx]])

        train_size = len(train_idx)
        train_prop = train_size / all_images
        val_size = len(val_idx)
        val_prop = val_size / all_images
        test_size = len(test_idx)
        test_prop = test_size / all_images
        
        train_size = int(train_prop * total_samples)
        steps_per_epoch = train_size // args.batch_size
        val_size = int(val_prop * total_samples)
        steps_per_epoch_val = val_size // args.batch_size
        test_size = int(test_prop * total_samples)
        steps_per_epoch_test = test_size // args.batch_size

    dataset_train, dataset_val, dataset_test = make_datasets(args.hdf5_path, batch_size=args.batch_size, T = args.pca_channels)

    with mlflow.start_run():
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_params({
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": 1e-5,
            "optimizer": "adam",
            "loss_fn": "SparseCategoricalCrossentropy",
            "train_size_images": len(train_idx),
            "val_size_images": len(val_idx),
            "test_size_images": len(test_idx),
        })

        if args.model_path == None:
            model = build_model()
            model.compile(
                optimizer=optimizers.Adam(1e-5),
                loss=losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"]
            )
        else:
            model = load_model(args.model_path)

        ckpt = callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)
        history = model.fit(
            dataset_train,
            validation_data=dataset_val,
            epochs=args.epochs,
            callbacks=[ckpt],
            steps_per_epoch = steps_per_epoch,
            validation_steps = steps_per_epoch_val
        )

        val_loss, val_acc = model.evaluate(dataset_test)
        mlflow.log_metric("val_loss", val_loss)
        mlflow.log_metric("val_acc", val_acc)
        model.save("final_model.keras")
        #mlflow.tensorflow.log_model(model, artifact_path="model")
        mlflow.tensorflow.log_model(
            model, 
            name="model"
        )

        with open("history.json", "w") as f:
            json.dump(history.history, f)
        mlflow.log_artifact("history.json")
