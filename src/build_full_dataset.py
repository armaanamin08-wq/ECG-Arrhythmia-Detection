import os
import wfdb
import numpy as np

# Folder containing the full MIT-BIH database
data_folder = "data/mitdb"

# Heartbeat annotation symbols we want to keep
beat_symbols = [
    "N", "L", "R", "A", "a", "J", "S",
    "V", "F", "e", "j", "E", "/"
]

# Storage for our full dataset
heartbeat_segments = []
heartbeat_labels = []
heartbeat_records = []
heartbeat_samples = []

# Find all ECG records by looking for .hea header files
record_names = sorted([
    filename.replace(".hea", "")
    for filename in os.listdir(data_folder)
    if filename.endswith(".hea")
])

print("Number of ECG records found:", len(record_names))
print("Records:", record_names)
# Define heartbeat window size
before_seconds = 0.3
after_seconds = 0.4

# Loop through every ECG record
for record_name in record_names:

    print("Processing record:", record_name)

    # Load ECG signal
    record = wfdb.rdrecord(
        os.path.join(data_folder, record_name)
    )

    # Load expert annotations
    annotation = wfdb.rdann(
        os.path.join(data_folder, record_name),
        "atr"
    )

    # Use the first ECG channel
    ecg_signal = record.p_signal[:, 0]

    before_samples = int(before_seconds * record.fs)
    after_samples = int(after_seconds * record.fs)

    # Go through every expert annotation
    for sample, symbol in zip(
        annotation.sample,
        annotation.symbol
    ):

        # Keep only actual heartbeat annotations
        if symbol not in beat_symbols:
            continue

        start = sample - before_samples
        end = sample + after_samples

        # Skip beats too close to the beginning or end
        if start < 0 or end > len(ecg_signal):
            continue

        segment = ecg_signal[start:end]

        # Store waveform, label, and record ID
        heartbeat_segments.append(segment)
        heartbeat_labels.append(symbol)
        heartbeat_records.append(record_name)
        heartbeat_samples.append(sample)

# Convert lists into NumPy arrays
heartbeat_segments = np.array(heartbeat_segments)
heartbeat_labels = np.array(heartbeat_labels)
heartbeat_records = np.array(heartbeat_records)
heartbeat_samples = np.array(heartbeat_samples)

print("\nFULL DATASET SUMMARY")
print("Number of heartbeat segments:", len(heartbeat_segments))
print("Dataset shape:", heartbeat_segments.shape)
print("Number of labels:", len(heartbeat_labels))
print("Number of record IDs:", len(heartbeat_records))
# Count heartbeat classes across the full database
unique_labels, label_counts = np.unique(
    heartbeat_labels,
    return_counts=True
)

print("\nFULL DATABASE HEARTBEAT DISTRIBUTION")

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
    # Group detailed MIT-BIH labels into broader heartbeat classes
def group_heartbeat_label(symbol):

    # Normal-type beats
    if symbol in ["N", "L", "R", "e", "j"]:
        return "N"

    # Supraventricular ectopic beats
    elif symbol in ["A", "a", "J", "S"]:
        return "S"

    # Ventricular ectopic beats
    elif symbol in ["V", "E"]:
        return "V"

    # Fusion beats
    elif symbol == "F":
        return "F"

    # Other / paced beats
    elif symbol == "/":
        return "Q"

    else:
        return "Q"


grouped_labels = np.array([
    group_heartbeat_label(symbol)
    for symbol in heartbeat_labels
])

# Count grouped classes
group_names, group_counts = np.unique(
    grouped_labels,
    return_counts=True
)

print("\nGROUPED HEARTBEAT DISTRIBUTION")

for group, count in zip(group_names, group_counts):

    percentage = count / len(grouped_labels) * 100

    print(
        group,
        ":",
        count,
        "beats |",
        round(percentage, 2),
        "%"
    )
    # Save the processed heartbeat dataset
output_file = "data/processed_heartbeat_dataset.npz"

np.savez_compressed(
    output_file,
    segments=heartbeat_segments,
    original_labels=heartbeat_labels,
    grouped_labels=grouped_labels,
    records=heartbeat_records,
    samples=heartbeat_samples
)

print("\nDataset saved successfully!")
print("Saved to:", output_file)