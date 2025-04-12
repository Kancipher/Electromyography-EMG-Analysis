import scipy.io
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def load_file(file_path):
    data = scipy.io.loadmat(file_path)
    subject = data['subject'][0][0]
    emg = data['emg']
    stimulus = data['stimulus'].flatten()
    return subject, emg, stimulus

def highpass_filter(data, cutoff=10, fs=100, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data, axis=0) 
    return filtered_data


def segment_trials(emg, stimulus, fs, break_label=0):
    trial_changes = np.where(np.diff(stimulus) != 0)[0] + 1
    trial_starts = np.concatenate(([0], trial_changes))
    trial_ends = np.concatenate((trial_changes, [len(stimulus)]))

    trials = []
    labels = []

    trial_lengths = []
    for start, end in zip(trial_starts, trial_ends):
        if stimulus[start] == break_label:
            continue
        trial_lengths.append(end - start)

    min_length = min(trial_lengths) if trial_lengths else 0

    for start, end in zip(trial_starts, trial_ends):
        trial_data = emg[start:end, :]
        trial_label = stimulus[start]

        if trial_label == break_label:
            continue

        if len(trial_data) >= min_length:
            trial_data = trial_data[:min_length]  
            trials.append(trial_data.flatten()) 
            labels.append(trial_label)

    return trials, labels


def leave_one_trial_out_cv(trials, labels, k_range):
    n_trials = len(trials)
    k_scores = {k: [] for k in k_range}

    X = np.array(trials)  
    y = np.array(labels)

    for k in k_range:
        trial_predictions = []
        trial_true_labels = []

        for test_idx in range(n_trials):
            train_mask = np.ones(n_trials, dtype=bool)
            train_mask[test_idx] = False

            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[~train_mask]
            y_test = y[~train_mask]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            knn = KNeighborsClassifier(n_neighbors=k)
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
    plt.yticks(np.arange(0,1.0,0.1))
    plt.show()


def find_best_configuration(emg, stimulus, fs, k_range=range(1, 20)):
    cutoff_frequency = 10  
    emg_filtered = highpass_filter(emg, cutoff=cutoff_frequency, fs=fs)

    best_accuracy = 0
    best_channel = None
    best_k = None
    num_channels = emg_filtered.shape[1]
    all_channel_results = []

    for channel in range(num_channels):
        print(f"\nTesting Channel {channel + 1}/{num_channels}")
        trials, labels = segment_trials(emg_filtered[:, channel].reshape(-1, 1), stimulus, fs)

        k_scores = leave_one_trial_out_cv(trials, labels, k_range)
        max_k, max_acc = BestKacc(k_scores)

        print(f"  => Channel {channel + 1} Results: Best K = {max_k}, Accuracy = {max_acc:.4f}")
        all_channel_results.append((channel + 1, max_k, max_acc))

        if max_acc > best_accuracy:
            best_accuracy = max_acc
            best_channel = channel
            best_k = max_k

        plot_k(k_scores, channel + 1)

    print(f"\nBest Overall Configuration:")
    print(f"  - Channel: {best_channel + 1}")
    print(f"  - Domain: High-Pass Filtered Data")
    print(f"  - Value of K: {best_k}")
    print(f"  - Highest Accuracy: {best_accuracy:.4f}")

    return best_channel, best_k, best_accuracy, all_channel_results

def main():
    fs = 100
    trial_duration = 5 
    break_label = 0
    file_path = "C://Users//Omar Ganna//Desktop//Project-1//subject_1.mat"
    subject, emg, stimulus = load_file(file_path)

    best_channel, best_k, best_accuracy, all_results = find_best_configuration(emg, stimulus, fs)

    trials, labels = segment_trials(emg, stimulus, fs, break_label)

    print(f"Number of Trials: {len(trials)}")

    print("\nSummary of All Channel Results:")
    for channel, k, acc in all_results:
        print(f"  - Channel {channel}: Best K = {k}, Accuracy = {acc:.4f}")

if __name__ == "__main__":
    main()