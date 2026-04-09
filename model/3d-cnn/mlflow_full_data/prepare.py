import argparse
import os
import zipfile
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from skimage.util import view_as_windows
import wget
import joblib
from sklearn.model_selection import train_test_split

ZIP_URLS = {
    "canola": "https://agdatacommons.nal.usda.gov/ndownloader/files/44757771",
    "kochia": "https://agdatacommons.nal.usda.gov/ndownloader/files/44735655",
    "ragweed": "https://agdatacommons.nal.usda.gov/ndownloader/files/44758256",
    "redroot_pigweed": "https://agdatacommons.nal.usda.gov/ndownloader/files/44742255",
    "soybean": "https://agdatacommons.nal.usda.gov/ndownloader/files/44758120",
    "sugarbeet": "https://agdatacommons.nal.usda.gov/ndownloader/files/44758033",
    "waterhemp": "https://agdatacommons.nal.usda.gov/ndownloader/files/44757875",
}

CLASS_MAPPING = {cls: i for i, cls in enumerate(ZIP_URLS.keys())}

def extract_patches(image, patch_size=(19, 19), stride=19):
    h, w, u = image.shape
    m, n = patch_size
    pad_h, pad_w = m // 2, n // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="reflect")
    patches = view_as_windows(padded_image, (m, n, u), step=stride)
    return patches.reshape(-1, m, n, u)
counter = 0
def process_zip(npy_path, scaler, pca, hdf5_file, class_id, T=70):
    global counter
    print(f"Processing {npy_path} for HDF5...")

    try:
        img = np.load(npy_path)
        H, W, B = img.shape
        print(f"Loaded {npy_path}, shape: {H}x{W}x{B}")

        flat = img.reshape(-1, B)
        flat_scaled = scaler.transform(flat)
        flat_reduced = pca.transform(flat_scaled)
        img_reduced = flat_reduced.reshape(H, W, T)
        print(f"Applied PCA, reduced shape: {H}x{W}x{T}")

        patches = extract_patches(img_reduced)
        if patches.size == 0:
            print(f"Warning: No patches extracted from {npy_path}")
            return
        print(f"Extracted {patches.shape[0]} patches from {npy_path}")

        labels = np.full(patches.shape[0], class_id, dtype=np.int32)

        ds_x = hdf5_file.create_dataset(f"X_{counter}", data=patches, dtype="float32")
        ds_y = hdf5_file.create_dataset(f"y_{counter}", data=labels, dtype="int32")
        print(f"Saved X_{counter} and y_{counter} to HDF5")
        counter += 1

    except Exception as e:
        print(f"Error processing {npy_path}: {e}")
        return

    if os.path.exists(npy_path):
        os.remove(npy_path)
        print(f"Removed {npy_path}")

def main(output_hdf5, n_components=70, max_files=2, proportion = (0.6,0.2,0.2)):
    data_dir = "/content/data"
    os.makedirs(data_dir, exist_ok=True)

    total_samples = max_files * 7

    train_proportion,val_proportion, test_proportion = proportion

    all_indices = list(range(total_samples))
    train_idx, val_test_idx = train_test_split(all_indices, train_size=train_proportion, random_state=42)

    test_size_proportion = test_proportion / (val_proportion + test_proportion)
    val_idx, test_idx = train_test_split(val_test_idx, test_size=test_size_proportion, random_state=42)


    scaler = StandardScaler()
    pca = IncrementalPCA(n_components=n_components)
    batch_size = 5000
    B = 224 
    train_counter = 0
    print("=== Pass 1: fitting scaler ===")
    for cls, url in ZIP_URLS.items():
        zip_path = f"{data_dir}/{cls}.zip"
        if not os.path.exists(zip_path):
            print(f"\nDownloading {cls} from {url}...")
            try:
                wget.download(url, zip_path)
                print(f"{cls} downloaded to {zip_path}")
            except Exception as e:
                print(f"Failed to download {cls}: {e}. Пропускаем.")
                continue

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                members = [m for m in zip_ref.namelist() if m.endswith(".npy")]
                for file_name in members[:max_files]:
                    print(f"Extracting {file_name}...")
                    zip_ref.extract(file_name, data_dir)
                    file_path = os.path.join(data_dir, file_name)

                    try:
                        img = np.load(file_path).reshape(-1, B)
                        # Fit scaler частями
                        if train_counter in train_idx:
                            for i in range(0, img.shape[0], batch_size):
                                scaler.partial_fit(img[i:i+batch_size])
                        train_counter += 1
                        print(f"Processed {file_name}")
                    except Exception as e:
                        print(f"Error processing {file_name}: {e}")
        except Exception as e:
            print(f"Error extracting from {zip_path}: {e}")
            continue

        if os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"Removed {zip_path}")

    print("=== Pass 2: fitting PCA ===")
    train_counter = 0
    for cls, _ in ZIP_URLS.items():
        cls_dir = f"{data_dir}/{cls}"
        npy_files = [f for f in os.listdir(cls_dir) if f.endswith(".npy") and cls in f]
        for file_name in npy_files[:max_files]:
            file_path = os.path.join(cls_dir, file_name)
            print(f"Processing {file_name} for PCA...")
            try:
                if train_counter in train_idx:
                    img = np.load(file_path).reshape(-1, B)
                    img_scaled = scaler.transform(img)
                    for i in range(0, img_scaled.shape[0], batch_size):
                        pca.partial_fit(img_scaled[i:i+batch_size])
                    print(f"PCA fitted on {file_name}")
                train_counter +=1
            except Exception as e:
                print(f"Error processing {file_name} for PCA: {e}")

    print("=== Pass 3: creating HDF5 ===")
    with h5py.File(output_hdf5, "w") as f:
        for cls, _ in ZIP_URLS.items():
            cls_dir = f"{data_dir}/{cls}"
            class_id = CLASS_MAPPING[cls]
            npy_files = [f for f in os.listdir(cls_dir) if f.endswith(".npy") and cls in f]
            for file_name in npy_files[:max_files]:
                file_path = os.path.join(cls_dir, file_name)
                print(f"Processing {file_name} for HDF5...")
                try:
                    process_zip(file_path, scaler, pca, f, class_id, T=n_components)
                    print(f"Patches from {file_name} added to HDF5")
                except Exception as e:
                    print(f"Error processing {file_name} for HDF5: {e}")
        f.attrs["train_indices"] = np.array(train_idx, dtype=np.int32)
        f.attrs["val_indices"] = np.array(val_idx, dtype=np.int32)
        f.attrs["test_indices"] = np.array(test_idx, dtype=np.int32)
        joblib.dump({"scaler": scaler, "pca": pca}, "scaler_pca.joblib")

    print(f"HDF5 dataset saved to {output_hdf5}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_hdf5", type=str, default="X_y_patches.h5")
    parser.add_argument("--n_components", type=int, default=70)
    parser.add_argument("--max_files", type=int, default=2)
    args = parser.parse_args()

    main(args.output_hdf5, args.n_components, args.max_files)
