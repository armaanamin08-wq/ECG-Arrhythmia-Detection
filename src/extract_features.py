import numpy as np

# Load processed heartbeat dataset
data = np.load("data/processed_heartbeat_dataset.npz")

segments = data["segments"]
original_labels = data["original_labels"]
grouped_labels = data["grouped_labels"]
records = data["records"]
samples = data["samples"]

print("Dataset loaded successfully!")
print("Heartbeat segments:", segments.shape)
print("Grouped labels:", grouped_labels.shape)
print("Records:", records.shape)

# Look at one heartbeat
first_beat = segments[0]

print("\nFirst heartbeat:")
print("Number of ECG measurements:", len(first_beat))
print("Minimum voltage:", round(np.min(first_beat), 3), "mV")
print("Maximum voltage:", round(np.max(first_beat), 3), "mV")
print("Voltage range:", round(np.ptp(first_beat), 3), "mV")
print("Heartbeat class:", grouped_labels[0])
print("Record:", records[0])
# Extract basic morphology features from every heartbeat
max_voltage = np.max(segments, axis=1)
min_voltage = np.min(segments, axis=1)
voltage_range = np.ptp(segments, axis=1)

# Root mean square (RMS) amplitude
rms_amplitude = np.sqrt(
    np.mean(segments ** 2, axis=1)
)

print("\nFEATURE EXTRACTION")
print("Number of max-voltage features:", len(max_voltage))
print("Number of voltage-range features:", len(voltage_range))
print("Number of RMS features:", len(rms_amplitude))

print("\nFirst heartbeat features:")
print("Maximum voltage:", round(max_voltage[0], 3), "mV")
print("Minimum voltage:", round(min_voltage[0], 3), "mV")
print("Voltage range:", round(voltage_range[0], 3), "mV")
print("RMS amplitude:", round(rms_amplitude[0], 3), "mV")
# -----------------------------------
# Beat timing features
# -----------------------------------

sampling_frequency = 360

previous_rr = np.full(len(samples), np.nan)
next_rr = np.full(len(samples), np.nan)

# Calculate RR intervals separately for each record
for record_name in np.unique(records):

    record_indices = np.where(records == record_name)[0]

    record_samples = samples[record_indices]

    rr_intervals = np.diff(record_samples) / sampling_frequency

    # Previous RR for beats 2 onward
    previous_rr[record_indices[1:]] = rr_intervals

    # Next RR for all beats except the last
    next_rr[record_indices[:-1]] = rr_intervals


print("\nRR TIMING FEATURES")

print("First 10 previous RR intervals:")
print(np.round(previous_rr[:10], 3))

print("\nFirst 10 next RR intervals:")
print(np.round(next_rr[:10], 3))
# Find the atrial premature beat in Record 100
a_indices = np.where(
    (records == "100") &
    (original_labels == "A")
)[0]

first_a_index = a_indices[0]

print("\nKNOWN ATRIAL PREMATURE BEAT")

print("Record:", records[first_a_index])
print("Original label:", original_labels[first_a_index])
print("Grouped class:", grouped_labels[first_a_index])
print("Sample:", samples[first_a_index])

print(
    "Previous RR:",
    round(previous_rr[first_a_index], 3),
    "seconds"
)

print(
    "Next RR:",
    round(next_rr[first_a_index], 3),
    "seconds"
)
# Calculate a relative RR timing feature
local_rr_average = (previous_rr + next_rr) / 2

rr_ratio = previous_rr / local_rr_average

print("\nRELATIVE RR FEATURE")

print(
    "A beat RR ratio:",
    round(rr_ratio[first_a_index], 3)
)
# -----------------------------------
# Waveform morphology area
# -----------------------------------

# Estimate each heartbeat's baseline using its median voltage
baseline = np.median(segments, axis=1)

# Subtract baseline from each heartbeat
baseline_corrected = segments - baseline[:, np.newaxis]

# Time between ECG samples
dt = 1 / sampling_frequency

# Calculate absolute waveform area
waveform_area = np.sum(
    np.abs(baseline_corrected),
    axis=1
) * dt

print("\nWAVEFORM AREA FEATURE")

print(
    "First heartbeat waveform area:",
    round(waveform_area[0], 4),
    "mV*s"
)
# -----------------------------------
# Compare features across heartbeat classes
# -----------------------------------

classes_to_compare = ["N", "S", "V", "F"]

print("\nCLASS FEATURE COMPARISON")

for heartbeat_class in classes_to_compare:

    class_indices = grouped_labels == heartbeat_class

    print("\nClass:", heartbeat_class)

    print(
        "Mean voltage range:",
        round(np.mean(voltage_range[class_indices]), 3),
        "mV"
    )

    print(
        "Mean RMS amplitude:",
        round(np.mean(rms_amplitude[class_indices]), 3),
        "mV"
    )

    print(
        "Mean waveform area:",
        round(np.mean(waveform_area[class_indices]), 4),
        "mV*s"
    )

    valid_rr = previous_rr[class_indices]
    valid_rr = valid_rr[~np.isnan(valid_rr)]

    print(
        "Mean previous RR:",
        round(np.mean(valid_rr), 3),
        "seconds"
    )