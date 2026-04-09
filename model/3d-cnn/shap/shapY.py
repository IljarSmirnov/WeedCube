import argparse
import math
import numpy as np
import h5py
from tensorflow.keras.models import load_model

def make_masked_ds(x, num, masks, side_ds):
    res = []

    for i in range(num):
        mask = masks[i]
        bg = side_ds[np.random.randint(len(side_ds))]

        x_masked = x.copy()

        for m in range(len(mask)):
            if mask[m] == 0:
                x_masked[0, :, :, m] = bg[:, :, m]

        res.append(x_masked)

    return np.vstack(res)

def prepare(file_path, image_num,T, M, objects):
    with h5py.File(file_path) as f:
        X=f[f"X_{image_num}"]
        g_ind=np.sort(np.random.choice(X.shape[0],objects,replace=False))
        X=X[g_ind]
    masks =[]
    mask_sizes = []

    masks.append(np.zeros(T))
    mask_sizes.append(0)
    masks.append(np.ones(T))
    mask_sizes.append(T)
    for i in range(T-1):
        tmp1=[0]*T
        tmp1[i]=1
        tmp2=[1]*T
        tmp2[i]=0
        masks.append(tmp1)
        masks.append(tmp2)
        mask_sizes.append(1)
        mask_sizes.append(T-1)
    for i in range(M-len(masks)):
        amount = np.random.choice(np.arange(1,T))
        idx = np.sort(np.random.choice(T,amount,replace=False))
        mask = np.zeros(T)
        mask[idx] = 1
        masks.append(mask)
        mask_sizes.append(sum(mask))

    masks = np.array(masks)
    print(masks.shape)
    with h5py.File(file_path) as f:
        test_idx = f.attrs["test_indices"]
        side_ds = []
        for i in test_idx:
            ds = f[f"X_{i}"]
            samples = ds[np.sort(np.random.choice(100, 40,replace=False))]
            side_ds.append(samples)
        side_ds = np.vstack(np.asarray(side_ds))
        print(side_ds.shape)
    return masks, mask_sizes, X, g_ind, side_ds


def main(model_path, file_path, M, image_num,objects):
    T=70
    model = load_model(model_path)
    masks, mask_sizes, X, g_ind, side_ds = prepare(file_path=file_path, image_num=image_num, T=T, M=M, objects=objects)
    weights = []
    for s in mask_sizes:
        if s == 0 or s == T:
            weights.append(1e6)
        else:
            weights.append((T-1) / ( math.comb(T, int(s)) * s * (T - s) ))
    W = np.array(weights)
    X_design =np.array(masks)
    n_classes = 7

    num = M
    preds_bl = model.predict(side_ds)
    for l in range(len(g_ind)):
        x=np.expand_dims(X[l],axis=0)
        masked_ds = make_masked_ds(x,num,masks,side_ds)
        preds = model.predict(masked_ds)
        pred_fx = model.predict(x)
        phi_by_class = np.zeros((n_classes,T))
        phi_all = []

        for c in range(n_classes):
            baseline = preds_bl[:,c].mean()
            fx = pred_fx[0,c]
            y = preds[:, c]-baseline
            delta_f = fx - baseline

            k = X_design.shape[1] - 1
            y_prime = y - X_design[:,k] * delta_f
            X_prime = X_design[:,:-1] - X_design[:,[k]]
            sqrt_w = np.sqrt(W)

            w = np.linalg.lstsq(
                sqrt_w[:,None]*X_prime,
                sqrt_w*y_prime,
                rcond=None
            )[0]

            phi = np.zeros(X_design.shape[1])
            phi[:-1] = w
            phi[-1] = delta_f - w.sum()
            phi_by_class[c] = phi
            print(delta_f, phi_by_class[c].sum())

        phi_all.append(phi_by_class)
    np.save(f"phi_mean_all_class_{image_num//10}.npy", phi_all)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="best_model.keras",required=True)
    parser.add_argument("--file_path", type=str, default="X_y_patches.h5",required=True)
    parser.add_argument("--M", type=int, default=12000)
    parser.add_argument("--image_num", type=int, default=60, required=True)
    parser.add_argument("--objects", type=int, default=40)
    args = parser.parse_args()

    main(args.model_path, args.file_path, args.M, args.image_num, args.objects, args.T)