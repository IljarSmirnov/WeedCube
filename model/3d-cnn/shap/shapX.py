import argparse
import math
import numpy as np
import h5py
from tensorflow.keras.models import load_model
import joblib
from skimage.util import view_as_windows

def make_masked_ds(x, num, masks, side_ds, scaler, pca):
    res = []
    bg = side_ds[np.random.randint(len(side_ds))] 
    for i in range(num):
        mask = masks[i]
        x_masked = x.copy()

        for m in range(len(mask)):
            if mask[m] == 0:
                x_masked[0, :, :, m] = bg[:, :, m]

        x_h,x_w,x_c = x_masked[0].shape
        corr_X = np.reshape(x_masked[0],(x_h*x_w,x_c))
        corr_X = scaler.transform(corr_X)
        corr_X = pca.transform(corr_X)
        new_x= np.expand_dims(np.reshape(corr_X,(x_h,x_w,70)),axis=0)
        res.append(new_x)

    return np.vstack(res)

def transform_patch(patch, scaler, pca):
    h,w,c = patch.shape
    tmp = patch.reshape(h*w,c)
    tmp = scaler.transform(tmp)
    tmp = pca.transform(tmp)
    return tmp.reshape(h,w,70)

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

def prepare(pca,scaler,data_test,imgs,T, M, objects,image_num = 10):
    path_to_img =  imgs[image_num]
    X = np.load(path_to_img)
    X = extract_patches(X)
    g_ind=np.sort(np.random.choice(X.shape[0],objects,replace=False)) #X.shape[0]
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

    side_ds = []
    for path in imgs:
        if path == path_to_img: continue
        img = np.load(path)
        ds = extract_patches(img)
        samples = ds[np.sort(np.random.choice(100, 30,replace=False))]
        side_ds.append(samples)
    if data_test is not None:
        side_ds = np.random.rand(20, 19, 19, T) 
    else:
        side_ds = np.vstack(np.asarray(side_ds))

    pca_side_ds = []
    for patch in side_ds:
        h,w,c = patch.shape
        corr = np.reshape(patch,(h*w,c))
        corr = scaler.transform(corr)
        corr = pca.transform(corr)
        corr = np.reshape(corr,(h,w,70))
        pca_side_ds.append(corr)
    pca_side_ds = np.array(pca_side_ds)
    return masks,mask_sizes,X,side_ds, pca_side_ds, g_ind


def main(imgs_path,M, image_num,objects,model_path,scaler_path,model_test = None, data_test = None):
    T=224
    with open(imgs_path) as f:
        imgs = f.read().splitlines()
    tmp_dict = joblib.load(scaler_path)
    pca = tmp_dict['pca']
    scaler = tmp_dict['scaler']
    model = load_model(model_path) 
    if model_test is not None:
        for layer in model.layers:
            weights = layer.get_weights()
            if weights:
                shuffled = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
                layer.set_weights(shuffled)
    masks, mask_sizes, X, side_ds, pca_side_ds, g_ind = prepare(pca,scaler,data_test,imgs,T,M,objects,image_num)

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

    preds_bl = model.predict(pca_side_ds)
    phi_all = []
    for l in range(len(g_ind)):
        if data_test is not None:
            x = np.random.rand(1, 19, 19, T) 
            x_pca = np.expand_dims(transform_patch(x[0], scaler, pca),axis=0)
        else:
            x=np.expand_dims(X[l],axis=0)
            x_pca = np.expand_dims(transform_patch(X[l], scaler, pca),axis=0)

        masked_ds = make_masked_ds(x,num,masks,side_ds,scaler,pca)
        preds = model.predict(masked_ds)
        pred_fx = model.predict(x_pca)
        phi_by_class = np.zeros((n_classes,T))

        print("turn ",l)
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

        phi_all.append(phi_by_class.copy())
    np.save(f"phi_all_class_{image_num//2}.npy", phi_all)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs_path", type=str, default="X_images.txt",required=True)
    parser.add_argument("--data_test", type=int, default=None)
    parser.add_argument("--model_test", type=int, default=None)
    parser.add_argument("--M", type=int, default=12000)
    parser.add_argument("--image_num", type=int, default=10,required=True)
    parser.add_argument("--objects", type=int, default=40)
    parser.add_argument("--model_path", type=str, default="/content/best_model.keras",required=True)
    parser.add_argument("--scaler_path", type=str, default="scaler_pca.joblib",required=True)
    args = parser.parse_args()

    main(
        imgs_path=args.imgs_path,
        M=args.M,
        image_num=args.image_num,
        objects=args.objects,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        model_test=args.model_test,
        data_test=args.data_test
    )