import scipy.io
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def load_file(file_path):
    data = scipy.io.loadmat(file_path)
    subject = data['subject'][0][0]
    emg = data['emg'] 
    stimulus = data['stimulus'].flatten()
    return subject, emg, stimulus


def highpass_filter(data, fs, cutoff=10, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data


def extract_time(signal):
    return signal.flatten()


def extract_frequency(signal):
    fft_values = np.fft.fft(signal, axis=0)
    fft_magnitude = np.abs(fft_values)
    half_idx = len(fft_magnitude) // 2
    fft_magnitude = fft_magnitude[:half_idx]
    return fft_magnitude.flatten()


def extract_combined(trial, type='combined'):
    if type == 'time':
        features = np.concatenate([extract_time(trial[:, i]) for i in range(trial.shape[1])])
    elif type == 'frequency':
        features = np.concatenate([extract_frequency(trial[:, i]) for i in range(trial.shape[1])])
    else:  
        time_features = np.concatenate([extract_time(trial[:, i]) for i in range(trial.shape[1])])
        freq_features = np.concatenate([extract_frequency(trial[:, i]) for i in range(trial.shape[1])])
        features = np.concatenate([time_features, freq_features])
    
    return features


def segment_trials(emg, stimulus, fs, duration_time):
    trial_length = fs * duration_time  
    trial_changes = np.where(np.diff(stimulus) != 0)[0] + 1
    trial_startpoint = np.concatenate(([0], trial_changes))
    trial_endpoint = np.concatenate((trial_changes, [len(stimulus)]))

    trials = []
    labels = []

    for start, end in zip(trial_startpoint, trial_endpoint):
        trial_data = emg[start:end, :]
        trial_label = stimulus[start]

        if trial_label == 0:
            continue

        if len(trial_data) >= trial_length:
            trial_data=trial_data[:trial_length]
        
        trials.append(trial_data)  
        labels.append(trial_label)

    return trials, labels


def leave_one_trial_out_cv_combined(trials, labels, k_range, type='combined'):
    n_trials = len(trials)
    k_scores = {k: [] for k in k_range}

    X = np.array([extract_combined(trial, type) for trial in trials])
    y = np.array(labels)

    for k in k_range:
        trial_predictions = []
        trial_true_labels = []

        for test_index in range(n_trials):
            train_mask = np.ones(n_trials, dtype=bool)
            train_mask[test_index] = False

            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[~train_mask]
            y_test = y[~train_mask]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            k_adjusted = min(k, len(X_train))

            knn = KNeighborsClassifier(n_neighbors=k_adjusted)
            knn.fit(X_train_scaled, y_train)
            pred = knn.predict(X_test_scaled)

            trial_predictions.extend(pred)
            trial_true_labels.extend(y_test)

        accuracy = accuracy_score(trial_true_labels, trial_predictions)
        k_scores[k].append(accuracy)
        print(f"  - K: {k}, Accuracy: {accuracy:.4f}")

    return k_scores


def BestKacc(k_scores):
    k_values = list(k_scores.keys())
    mean_scores = [np.mean(scores) for scores in k_scores.values()]
    best_k = k_values[np.argmax(mean_scores)]
    best_accuracy = max(mean_scores)
    return best_k, best_accuracy


def plot_k(k_scores, channel):
    k_values = list(k_scores.keys())
    mean_scores = [np.mean(scores) for scores in k_scores.values()]

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, mean_scores, 'bo-')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Classification Accuracy')
    plt.title(f'KNN Performance vs K Value (Channel {channel})')
    plt.grid(True)
    plt.xticks(np.arange(1,20))
    plt.yticks(np.arange(0,1.1,0.1))
    plt.show()



def find_best_combined_configuration(emg, stimulus, fs, k_range=range(1, 20), duration_time=5):
    print("\nTesting Time Domain Features")
    filtered_data = highpass_filter(emg, fs)
    trials, labels = segment_trials(filtered_data, stimulus, fs, duration_time)

    k_scores_time = leave_one_trial_out_cv_combined(trials, labels, k_range, type='time')
    best_k_time, best_accuracy_time = BestKacc(k_scores_time)
    print(f"\nBest Time Features Configuration: K = {best_k_time}, Accuracy = {best_accuracy_time:.4f}")
    plot_k(k_scores_time, "Time Features")

    print("\nTesting Frequency Domain Features")
    k_scores_freq = leave_one_trial_out_cv_combined(trials, labels, k_range, type='frequency')
    best_k_freq, best_accuracy_freq = BestKacc(k_scores_freq)
    print(f"\nBest Frequency Features Configuration: K = {best_k_freq}, Accuracy = {best_accuracy_freq:.4f}")
    plot_k(k_scores_freq, "Frequency Features")

    k_scores_comb = leave_one_trial_out_cv_combined(trials, labels, k_range)
    best_k_comb, best_accuracy_comb = BestKacc(k_scores_comb)
    print(f"\nBest Combined time and frequency Features Configuration: K = {best_k_comb}, Accuracy = {best_accuracy_comb:.4f}")
    plot_k(k_scores_comb, "Combined time and frequency Features")
    return best_k_time, best_accuracy_time, best_k_freq, best_accuracy_freq, best_k_comb, best_accuracy_comb


def main():
    fs = 100
    file_path = "C://Users//Omar Ganna//Desktop//Project-1//subject_1.mat"
    subject, emg, stimulus = load_file(file_path)

    print(f"Processing Subject {subject}")

    best_k_time, best_accuracy_time, best_k_freq, best_accuracy_freq, best_k_comb, best_accuracy_comb = find_best_combined_configuration(emg, stimulus, fs)

    print("\nSummary:")
    print(f"  - Time Features: Best K = {best_k_time}, Accuracy = {best_accuracy_time:.4f}")
    print(f"  - Frequency Features: Best K = {best_k_freq}, Accuracy = {best_accuracy_freq:.4f}")
    print(f"  - Combined Features: Best K = {best_k_comb}, Accuracy = {best_accuracy_comb:.4f}")


if __name__ == "__main__":
    main()
