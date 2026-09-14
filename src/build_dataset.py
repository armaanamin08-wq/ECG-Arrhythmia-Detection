import wfdb
import numpy as np
import matplotlib.pyplot as plt

# Load ECG record and expert annotations
record = wfdb.rdrecord("data/100")
annotation = wfdb.rdann("data/100", "atr")

# Use the MLII channel
ecg_signal = record.p_signal[:, 0]

# Heartbeat symbols we want to keep
beat_symbols = [
    "N", "L", "R", "A", "a", "J", "S",
    "V", "F", "e", "j", "E", "/"
]

# Window around each heartbeat
before_seconds = 0.3
after_seconds = 0.4

before_samples = int(before_seconds * record.fs)
after_samples = int(after_seconds * record.fs)

# Store heartbeat segments and labels here
heartbeat_segments = []
heartbeat_labels = []

for sample, symbol in zip(annotation.sample, annotation.symbol):

    # Skip annotations that are not actual heartbeats
    if symbol not in beat_symbols:
        continue

    start = sample - before_samples
    end = sample + after_samples

    # Skip beats too close to the beginning or end of the recording
    if start < 0 or end > len(ecg_signal):
        continue

    segment = ecg_signal[start:end]

    heartbeat_segments.append(segment)
    heartbeat_labels.append(symbol)

heartbeat_segments = np.array(heartbeat_segments)
heartbeat_labels = np.array(heartbeat_labels)

print("Number of heartbeat segments:", len(heartbeat_segments))
print("Shape of heartbeat dataset:", heartbeat_segments.shape)
print("First 20 labels:", heartbeat_labels[:20])
# Count how many heartbeats belong to each class
unique_labels, label_counts = np.unique(
    heartbeat_labels,
    return_counts=True
)

print("\nHeartbeat Type Distribution")

for label, count in zip(unique_labels, label_counts):
    percentage = count / len(heartbeat_labels) * 100

    print(
        label,
        ":",
        count,
        "beats |",
        round(percentage, 2),
        "%"
    )
    # Plot heartbeat class distribution
plt.figure(figsize=(10, 5))

plt.bar(unique_labels, label_counts)

plt.xlabel("Heartbeat Type")
plt.ylabel("Number of Beats")
plt.title("Heartbeat Class Distribution - MIT-BIH Record 100")

plt.grid(axis="y")

plt.show()