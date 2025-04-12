import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

def Loading_file(file_path):
    data = scipy.io.loadmat(file_path)
    subject = data['subject'][0][0]
    emg = data['emg']
    stimulus = data['stimulus'].flatten()
    return subject, emg, stimulus

def highpass_filter(data, cutoff=10, fs=100, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = lfilter(b, a, data, axis=0)  
    return filtered_data


def Time_domain_features(signal):
    mav = np.mean(np.abs(signal), axis=0)
    rms = np.sqrt(np.mean(signal ** 2, axis=0))
    var = np.var(signal, axis=0)  
    zero_crossings = np.sum(np.diff(np.sign(signal), axis=0) != 0, axis=0)
    return {
        "MAV": mav,
        "RMS": rms,
        "Variance": var,
        "Zero_Crossings": zero_crossings
    }

if __name__ == "__main__":
    file_path = "C://Users//Omar Ganna//Desktop//Project-1//subject_1.mat"
    subject, emg, stimulus = Loading_file(file_path)

    print(f"Subject: {subject}")
    print(f"EMG shape: {emg.shape}")
    print(f"Stimulus shape: {stimulus.shape}")

    fs = 100
    filtered_emg = highpass_filter(emg, cutoff=10, fs=fs)

    for i in range(3):  
        plt.figure(figsize=(10, 6))

        y_min = min(np.min(emg[:, i]), np.min(filtered_emg[:, i]))
        y_max = max(np.max(emg[:, i]), np.max(filtered_emg[:, i]))

        plt.subplot(2, 1, 1)  
        plt.plot(emg[:, i], label=f'Original Signal - Channel {i + 1}', alpha=0.7)
        plt.title(f"Original Signal - Channel {i + 1}")
        plt.xlabel("Sample Index")
        plt.ylabel("Signal Amplitude")
        plt.legend()
        plt.ylim(y_min, y_max)  

        plt.subplot(2, 1, 2)  
        plt.plot(filtered_emg[:, i], label=f'Filtered Signal - Channel {i + 1}', alpha=0.7, color='orange')  
        plt.title(f"Filtered Signal - Channel {i + 1}")
        plt.xlabel("Sample Index")
        plt.ylabel("Signal Amplitude")
        plt.legend()
        plt.ylim(y_min, y_max)  

        plt.tight_layout()
        plt.show()

    features = Time_domain_features(filtered_emg)
    print("Time-Domain Features for Each Channel:")
    for feature_name, values in features.items():
        print(f"{feature_name}: {values}")
