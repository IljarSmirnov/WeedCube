from tensorflow.keras.models import Model, load_model
import argparse
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
import seaborn as sns
import utils
import prepare
import numpy as np
import matplotlib.pyplot as plt
import joblib
def get_probs_and_labels(model,data_path):
    _,_, test_ds = utils.make_datasets(data_path)
    probs = model.predict(test_ds)

    true_labels = []
    for patches, labels in test_ds:
        true_labels.extend(labels.numpy())
    true_labels = np.array(true_labels)

    return true_labels,probs

def print_confusion_matr(preds, true_labels):
    cm = confusion_matrix(true_labels, preds, normalize='true')
    sns.heatmap(cm, annot=True, cmap='coolwarm', fmt=".2f")
    plt.savefig("confusion_matrix.png")


def print_report(preds,true_labels):
    print(classification_report(true_labels, preds))

def plot_distrib_of_model_confidence(probs):
    confidences = np.max(probs, axis=1)  
    preds = np.argmax(probs, axis=1)
    correct_mask = preds == true_labels
    wrong_mask = ~correct_mask

    conf_correct = confidences[correct_mask]
    conf_wrong = confidences[wrong_mask]
    plt.hist(conf_correct, bins=30, alpha=0.6, label='Correct')
    plt.xlabel('Model confidence')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Distribution of Model Confidence')
    plt.savefig("correct_conf.png")
    plt.show()
    plt.hist(conf_wrong, bins=30, alpha=0.6, label='Wrong')
    plt.xlabel('Model confidence')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Distribution of Model Confidence')
    plt.savefig("wrong_conf.png")
    plt.show()

def plot_calib(probs):
    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    correct = (preds == true_labels).astype(int)

    prob_true, prob_pred = calibration_curve(correct, confidences, n_bins=10)

    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
    plt.xlabel('Predicted confidence')
    plt.ylabel('True accuracy')
    plt.title('Calibration curve (Reliability diagram)')
    plt.legend()
    plt.grid()
    plt.savefig("calib_curve.png")
    plt.show()

def display_errors(imag_path, model, scaler, pca):
    img = np.load(imag_path)
    band_to_display = img[:, :, img.shape[2] // 2]
    plt.imshow(band_to_display, cmap='gray')
    plt.title('Grayscale Image from HSI Data')
    plt.axis('off')
    plt.show()

    r,c,b = img.shape
    img = img.reshape((r*c,b))
    img = scaler.transform(img)
    img = pca.transform(img)
    img = img.reshape((r,c,70))
    patches = prepare.extract_patches(img)
    vpreds = model.predict(patches)
    vpreds = np.argmax(vpreds,axis=1)

    gray = band_to_display
    m, n = (19,19)
    pad_h = m // 2
    pad_w = n // 2
    gray = np.pad(gray, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    H,W = gray.shape

    counter = 0
    gray = gray.astype(np.int64)
    counter = 0
    for i in range(0, H - 19 + 1, 14):
        for j in range(0, W - 19 + 1, 14):
            if counter >= len(vpreds):
                break
            gray[i:i+19, j:j+19] = vpreds[counter] * 36
            counter += 1

    gray = np.clip(gray,0,255)
    gray = gray.astype(np.int64)
    plt.imshow(gray,cmap="gray")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_hdf5", type=str, default="X_y_patches.h5")
    parser.add_argument("--model_path", type=str, default="model.keras")
    parser.add_argument("--funcs", type = str, choices=["confusion_matr", "classif_report", "distrib_conf", "calib", "display_errors"], nargs='+')
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--scaler_pca_path", type=str)
    args = parser.parse_args()
    model= load_model(args.model_path)
    true_labels,probs = get_probs_and_labels(model, args.data_hdf5)
    if "confusion_matr" in args.funcs:
        print_confusion_matr(np.argmax(probs,axis=1),true_labels)
    if "classif_report" in args.funcs:
        print_report(np.argmax(probs,axis=1),true_labels)
    if "distrib_conf" in args.funcs:
        plot_distrib_of_model_confidence(probs)
    if "calib" in args.funcs:
        plot_calib(probs)
    if "display_errors" in args.funcs:
        data = joblib.load(args.scaler_pca_path)
        scaler,pca = data["scaler"], data["pca"]
        display_errors(args.image_path, model, scaler, pca)