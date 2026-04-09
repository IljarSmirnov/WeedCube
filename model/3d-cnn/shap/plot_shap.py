import joblib
from scipy.signal import find_peaks
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt

def plot_shap_vectors(filename="phi_all_class_{}.npy", type = "X"):
    if type == "Y":
        d=joblib.load("scaler_pca.joblib")
        pca=d["pca"]
        pca_w=pca.components_
        pca_w2=pca_w**2

    peaks_by_class_sg=[]
    peaks_by_class_raw=[]
    for i in range(7):
        phi_all = np.load(filename.format(i))
        if type == "X":
            phi_mean_all = np.abs(phi_all)
            phi_mean_all=phi_mean_all.mean(axis=0)
            abs_shap = phi_mean_all[i]
        elif type == "Y":
            phi_for_current_class = phi_all[:, i, :]
            raw_shap = phi_for_current_class @ pca_w2
            abs_raw_shap = np.abs(raw_shap) 
            abs_shap = abs_raw_shap.mean(axis=0)

        smoothed_shap = savgol_filter(abs_shap, window_length=7, polyorder=3)
        threshold_prominence = 0.1 * np.max(smoothed_shap)
        peaks, properties = find_peaks(smoothed_shap, prominence=threshold_prominence)


        plt.figure(figsize=(10, 6)) 
        plt.subplot(2, 1, 2) 
        plt.plot(smoothed_shap, label='Smoothed |SHAP|', color='blue')
        plt.plot(peaks, smoothed_shap[peaks], "x", color='red', markersize=10, label='Peaks')
        plt.title(f"The importance of channels for the class {i}",fontsize = 18)
        plt.xlabel("Physical channel",fontsize = 14)
        plt.ylabel("Importance (SHAP)",fontsize = 14)
        plt.legend(fontsize = 12)
        plt.grid(True)
        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)


        sorted_peak_indices = np.argsort(smoothed_shap[peaks])[::-1]
        sorted_peaks_by_shap = peaks[sorted_peak_indices]
        peaks_by_class_sg.append(sorted_peaks_by_shap)

        num = len(sorted_peaks_by_shap)
        descending_indices_abs_shap = np.argsort(abs_shap)[::-1]
        peaks_by_class_raw.append(descending_indices_abs_shap[:num])

        plt.subplot(2, 1, 1)
        plt.plot(abs_shap, label='Raw |SHAP|', marker = '.',color='blue')
        plt.plot(descending_indices_abs_shap[:num], abs_shap[descending_indices_abs_shap[:num]], "x", color='red', markersize=10, label='Peaks')
        plt.title(f"The importance of channels for the class {i}",fontsize = 18)
        plt.xlabel("Physical channel",fontsize = 14)
        plt.ylabel("Importance (SHAP)",fontsize = 14)
        plt.legend(fontsize = 12)
        plt.grid(True)
        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)
        plt.tight_layout()
        plt.savefig('shap_graph.png', format='png', bbox_inches='tight')
        plt.show()
    return peaks_by_class_sg, peaks_by_class_raw
def plot_peaks_distib(peaks_by_class_sg, peaks_by_class_raw):
    plt.figure(figsize = (10,4))
    plt.subplot(1,2,1)
    for i in range(len(peaks_by_class_sg)):
        plt.plot([i]*len(peaks_by_class_sg[i]),peaks_by_class_sg[i],marker='o',linestyle=" ",alpha=0.7)

    plt.subplot(1,2,2)
    for i in range(len(peaks_by_class_raw)):
        plt.plot([i]*len(peaks_by_class_raw[i]),peaks_by_class_raw[i],marker='o',linestyle=" ",alpha=0.7)
    plt.savefig("X_shap.pdf",format = "pdf", bbox_inches = 'tight')
    plt.show()

def main(type="X"):
    peaks_by_class_sg, peaks_by_class_raw = plot_shap_vectors(type=type)
    plot_peaks_distib(peaks_by_class_sg, peaks_by_class_raw)
