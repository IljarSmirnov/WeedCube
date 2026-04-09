import numpy as np
from sklearn.metrics import accuracy_score
import joblib
from tensorflow.keras.models import load_model
from skimage.util import view_as_windows
import gc
import argparse

def extract_patches(image, patch_size=(19, 19), stride=14):
    h, w, u = image.shape
    m, n = patch_size
    if m > h or n > w:
        raise ValueError("Patch size is larger than image dimensions")
    pad_h = m // 2
    pad_w = n // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='reflect')
    patches = view_as_windows(padded_image, (m, n, u), step=stride)
    patches = patches.reshape(-1, m, n, u)
    return patches

def main(model_path, scaler_pca_path, imgs_path, ablation_channels=50):
    model = load_model(model_path)
    tmp_dict = joblib.load(scaler_pca_path)
    pca = tmp_dict['pca']
    scaler = tmp_dict['scaler']

    class_mapping = {
        "canola": 0,
        "kochia": 1,
        "ragweed": 2,
        "redroot": 3,
        "soybean": 4,
        "sugarbeet": 5,
        "waterhemp": 6,
    }

    with open(imgs_path) as f:
        imgs = f.read().splitlines()

    # Base accuracy calculation
    y_true_base_list = []
    y_pred_base_list = []

    print("Calculating base accuracy...")
    for path in imgs:
        img = np.load(path)
        h, w, c = img.shape[0], img.shape[1], img.shape[2]
        img_reshaped = np.reshape(img, (h * w, c))
        img_scaled = scaler.transform(img_reshaped)
        img_pca = pca.transform(img_scaled)
        img_processed = np.reshape(img_pca, (h, w, 70))
        patches = extract_patches(img_processed)

        current_preds = np.argmax(model.predict(patches), axis=1)
        y_pred_base_list.append(current_preds)

        current_labels = np.array([class_mapping[path.split("/")[5]]] * patches.shape[0])
        y_true_base_list.append(current_labels)

        del img, img_reshaped, img_scaled, img_pca, img_processed, patches, current_preds, current_labels
        gc.collect()

    y_true_base = np.concatenate(y_true_base_list, axis=0)
    y_pred_base = np.concatenate(y_pred_base_list, axis=0)
    base_score = accuracy_score(y_true_base, y_pred_base)

    print(f"Base accuracy (all channels): {base_score:.4f}")

    # Ablated accuracy calculation
    y_true_ablated_list = []
    y_pred_ablated_list = []

    print("Calculating ablated accuracy...")
    for path in imgs:
        img = np.load(path)
        h, w, c = img.shape[0], img.shape[1], img.shape[2]
        img_reshaped = np.reshape(img, (h * w, c))
        img_scaled = scaler.transform(img_reshaped)
        img_scaled[:, ablation_channels:] = 0 
        img_pca = pca.transform(img_scaled)
        img_processed = np.reshape(img_pca, (h, w, 70))
        patches = extract_patches(img_processed)

        current_preds_ablated = np.argmax(model.predict(patches), axis=1)
        y_pred_ablated_list.append(current_preds_ablated)

        current_labels = np.array([class_mapping[path.split("/")[5]]] * patches.shape[0])
        y_true_ablated_list.append(current_labels)

        del img, img_reshaped, img_scaled, img_pca, img_processed, patches, current_preds_ablated, current_labels
        gc.collect()

    y_true_ablated = np.concatenate(y_true_ablated_list, axis=0)
    y_pred_ablated = np.concatenate(y_pred_ablated_list, axis=0)
    ablated_score = accuracy_score(y_true_ablated, y_pred_ablated)

    print(f"Accuracy only on the first {ablation_channels} channels: {ablated_score:.4f}")
    print(f"Quality drop: {(base_score - ablated_score) * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Keras model file.")
    parser.add_argument("--scaler_pca_path", type=str, required=True, help="Path to the scaler and PCA joblib file.")
    parser.add_argument("--imgs_path", type=str, required=True, help="Path to the text file containing image paths.")
    parser.add_argument("--ablation_channels", type=int, default=50, help="Number of channels to keep (ablate from this onwards). Default: 50.")

    args = parser.parse_args()
    main(args.model_path, args.scaler_pca_path, args.imgs_path, args.ablation_channels)