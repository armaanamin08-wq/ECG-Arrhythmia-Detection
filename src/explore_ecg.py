import matplotlib
matplotlib.use("TkAgg")

import wfdb
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
# Load ECG record 100 from our data folder
record = wfdb.rdrecord("data/100")

# Display basic information about the ECG
print("Sampling frequency:", record.fs, "Hz")
print("Number of samples:", record.sig_len)
print("Signal names:", record.sig_name)
print("Number of channels:", record.n_sig)
# Extract the first 10 seconds from the MLII channel
duration = 10
samples = duration * record.fs

ecg_signal = record.p_signal[:int(samples), 0]

# Create a time axis in seconds
time = [i / record.fs for i in range(len(ecg_signal))]

# Plot the ECG
plt.figure(figsize=(12, 4))
plt.plot(time, ecg_signal)

plt.xlabel("Time (seconds)")
plt.ylabel("ECG Amplitude (mV)")
plt.title("MIT-BIH Record 100 - MLII ECG (First 10 Seconds)")
plt.grid(True)

plt.show()

# Clean the ECG signal to reduce noise
cleaned_ecg = nk.ecg_clean(
    ecg_signal,
    sampling_rate=record.fs,
    method="neurokit"
)

# Compare the raw and cleaned signals
plt.figure(figsize=(12, 5))

plt.plot(time, ecg_signal, label="Raw ECG", alpha=0.6)
plt.plot(time, cleaned_ecg, label="Cleaned ECG")

plt.xlabel("Time (seconds)")
plt.ylabel("ECG Amplitude")
plt.title("Raw ECG vs Cleaned ECG")
plt.legend()
plt.grid(True)

plt.show()
# Detect R-peaks in the cleaned ECG
signals, info = nk.ecg_peaks(
    cleaned_ecg,
    sampling_rate=record.fs
)

r_peaks = info["ECG_R_Peaks"]

print("Number of R-peaks detected:", len(r_peaks))
print("R-peak sample locations:", r_peaks)

# Convert R-peak locations from samples to seconds
r_peak_times = r_peaks / record.fs

print("R-peak times (seconds):", r_peak_times)
import numpy as np

# Calculate the time between consecutive R-peaks
rr_intervals = np.diff(r_peak_times)

# Calculate heart rate for each interval
heart_rates = 60 / rr_intervals

# Calculate average heart rate
average_heart_rate = np.mean(heart_rates)

print("RR intervals (seconds):", rr_intervals)
print("Heart rates (BPM):", heart_rates)
print("Average heart rate:", round(average_heart_rate, 2), "BPM")
# Plot cleaned ECG with detected R-peaks
plt.figure(figsize=(12, 5))

plt.plot(time, cleaned_ecg, label="Cleaned ECG")

# Mark each detected R-peak
plt.scatter(
    r_peak_times,
    cleaned_ecg[r_peaks],
    marker="o",
    label="Detected R-peaks"
)

plt.xlabel("Time (seconds)")
plt.ylabel("ECG Amplitude")
plt.title("Automatic R-Peak Detection")
plt.legend()
plt.grid(True)

plt.show()
# Load expert heartbeat annotations
annotation = wfdb.rdann("data/100", "atr")

# Get annotation locations in samples
expert_samples = annotation.sample

# Keep only expert annotations from the first 10 seconds
expert_samples_10s = expert_samples[expert_samples < samples]

# Convert expert sample locations to seconds
expert_times_10s = expert_samples_10s / record.fs

print("Expert annotation sample locations:", expert_samples_10s)
print("Expert annotation times (seconds):", expert_times_10s)
print("Number of expert annotations:", len(expert_samples_10s))
# Get the expert annotation symbols
expert_symbols = annotation.symbol

# Show annotations from the first 10 seconds
expert_symbols_10s = [
    expert_symbols[i]
    for i in range(len(expert_samples))
    if expert_samples[i] < samples
]

print("Expert annotation symbols:", expert_symbols_10s)

# Print each annotation in an easy-to-read format
print("\nExpert annotations in first 10 seconds:")

for sample, symbol in zip(expert_samples_10s, expert_symbols_10s):
    time_seconds = sample / record.fs

    print(
        "Time:",
        round(time_seconds, 3),
        "seconds | Sample:",
        sample,
        "| Symbol:",
        symbol
    )
    # Define symbols that represent actual heartbeats
beat_symbols = [
    "N", "L", "R", "A", "a", "J", "S",
    "V", "F", "e", "j", "E", "/"
]

# Keep only actual heartbeat annotations
true_beat_samples = []

for sample, symbol in zip(expert_samples_10s, expert_symbols_10s):
    if symbol in beat_symbols:
        true_beat_samples.append(sample)

true_beat_samples = np.array(true_beat_samples)

print("\nNumber of expert-labeled heartbeats:", len(true_beat_samples))
print("Expert heartbeat locations:", true_beat_samples)
# Compare detected R-peaks to expert-labeled heartbeats

tolerance_seconds = 0.05
tolerance_samples = int(tolerance_seconds * record.fs)

true_positives = 0
matched_expert_beats = set()

for detected_peak in r_peaks:
    for i, expert_peak in enumerate(true_beat_samples):

        if i in matched_expert_beats:
            continue

        if abs(detected_peak - expert_peak) <= tolerance_samples:
            true_positives += 1
            matched_expert_beats.add(i)
            break

false_positives = len(r_peaks) - true_positives
false_negatives = len(true_beat_samples) - true_positives

print("\nR-Peak Detection Validation")
print("True Positives:", true_positives)
print("False Positives:", false_positives)
print("False Negatives:", false_negatives)
# Calculate R-peak detection sensitivity
sensitivity = true_positives / (true_positives + false_negatives) * 100

print("Sensitivity:", round(sensitivity, 2), "%")
# Calculate positive predictive value
ppv = true_positives / (true_positives + false_positives) * 100

print("Positive Predictive Value:", round(ppv, 2), "%")