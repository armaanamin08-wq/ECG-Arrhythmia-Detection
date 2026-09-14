import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)
from sklearn.model_selection import GroupShuffleSplit
from scipy.signal import butter, sosfiltfilt

# -----------------------------------
# Load processed ECG dataset
# -----------------------------------

data = np.load(
    "data/processed_heartbeat_dataset.npz"
)

segments = data["segments"]
labels = data["grouped_labels"]
records = data["records"]
samples = data["samples"]

sampling_frequency = 360
# -----------------------------------
# Calculate RR timing on FULL dataset
# before excluding any classes
# -----------------------------------

full_previous_rr = np.full(
    len(samples),
    np.nan
)

full_next_rr = np.full(
    len(samples),
    np.nan
)

for record_name in np.unique(records):

    record_indices = np.where(
        records == record_name
    )[0]

    record_samples = samples[
        record_indices
    ]

    rr_intervals = (
        np.diff(record_samples)
        / sampling_frequency
    )

    full_previous_rr[
        record_indices[1:]
    ] = rr_intervals

    full_next_rr[
        record_indices[:-1]
    ] = rr_intervals


full_local_rr_average = (
    full_previous_rr
    + full_next_rr
) / 2

full_rr_ratio = (
    full_previous_rr
    / full_local_rr_average
)

full_previous_rr = np.nan_to_num(
    full_previous_rr,
    nan=0.0
)

full_next_rr = np.nan_to_num(
    full_next_rr,
    nan=0.0
)

full_rr_ratio = np.nan_to_num(
    full_rr_ratio,
    nan=1.0
)


print("Dataset loaded!")
print("Heartbeat segments:", segments.shape)
print("Number of records:", len(np.unique(records)))


# -----------------------------------
# Select example normal heartbeat
# -----------------------------------

normal_indices = np.where(
    labels == "N"
)[0]

example_index = normal_indices[100]

clean_beat = segments[
    example_index
].copy()

example_record = records[
    example_index
]

time = (
    np.arange(len(clean_beat))
    / sampling_frequency
    - 0.3
)


print("\nEXAMPLE HEARTBEAT")
print("Class:", labels[example_index])
print("Record:", example_record)
print("Measurements:", len(clean_beat))
print(
    "Duration:",
    round(
        len(clean_beat)
        / sampling_frequency,
        3
    ),
    "seconds"
)


# -----------------------------------
# Plot original heartbeat
# -----------------------------------

plt.figure(
    figsize=(10, 4)
)

plt.plot(
    time,
    clean_beat,
    label="Original ECG"
)

plt.axvline(
    0,
    linestyle="--",
    label="Expert beat annotation"
)

plt.xlabel(
    "Time relative to annotation (s)"
)

plt.ylabel(
    "ECG voltage (mV)"
)

plt.title(
    "Original ECG Heartbeat Before Noise Testing"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/original_heartbeat_noise_test.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# -----------------------------------
# Simulate baseline wander
# -----------------------------------

baseline_frequency = 0.5
baseline_amplitude = 0.20

baseline_wander = (
    baseline_amplitude
    * np.sin(
        2
        * np.pi
        * baseline_frequency
        * time
    )
)

baseline_corrupted_beat = (
    clean_beat
    + baseline_wander
)


# -----------------------------------
# Plot original vs baseline wander
# -----------------------------------

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    time,
    clean_beat,
    label="Original ECG"
)

plt.plot(
    time,
    baseline_corrupted_beat,
    label="ECG + baseline wander"
)

plt.xlabel(
    "Time relative to annotation (s)"
)

plt.ylabel(
    "ECG voltage (mV)"
)

plt.title(
    "Effect of Simulated Baseline Wander on ECG"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/baseline_wander_example.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# -----------------------------------
# Quantify baseline-wander distortion
# -----------------------------------

signal_power = np.mean(
    clean_beat ** 2
)

noise_power = np.mean(
    baseline_wander ** 2
)

snr_db = 10 * np.log10(
    signal_power / noise_power
)

rmse = np.sqrt(
    np.mean(
        (
            baseline_corrupted_beat
            - clean_beat
        ) ** 2
    )
)

print("\nBASELINE WANDER SIGNAL QUALITY")

print(
    "Signal power:",
    round(signal_power, 5)
)

print(
    "Noise power:",
    round(noise_power, 5)
)

print(
    "SNR:",
    round(snr_db, 2),
    "dB"
)

print(
    "RMSE:",
    round(rmse, 4),
    "mV"
)
# -----------------------------------
# Simulate random measurement noise
# -----------------------------------

np.random.seed(42)

random_noise_std = 0.10

random_noise = np.random.normal(
    loc=0.0,
    scale=random_noise_std,
    size=len(clean_beat)
)

random_corrupted_beat = (
    clean_beat
    + random_noise
)


# -----------------------------------
# Quantify random-noise distortion
# -----------------------------------

random_noise_power = np.mean(
    random_noise ** 2
)

random_snr_db = 10 * np.log10(
    signal_power / random_noise_power
)

random_rmse = np.sqrt(
    np.mean(
        (
            random_corrupted_beat
            - clean_beat
        ) ** 2
    )
)


print("\nRANDOM NOISE SIGNAL QUALITY")

print(
    "Noise standard deviation:",
    random_noise_std,
    "mV"
)

print(
    "Noise power:",
    round(random_noise_power, 5)
)

print(
    "SNR:",
    round(random_snr_db, 2),
    "dB"
)

print(
    "RMSE:",
    round(random_rmse, 4),
    "mV"
)


# -----------------------------------
# Plot original vs random-noise ECG
# -----------------------------------

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    time,
    clean_beat,
    label="Original ECG"
)

plt.plot(
    time,
    random_corrupted_beat,
    label="ECG + random noise"
)

plt.xlabel(
    "Time relative to annotation (s)"
)

plt.ylabel(
    "ECG voltage (mV)"
)

plt.title(
    "Effect of Simulated Random Noise on ECG"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/random_noise_example.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# -----------------------------------
# Recreate feature extraction function
# -----------------------------------
def bandpass_filter_ecg(
    signal,
    sampling_rate=360,
    low_cutoff=0.5,
    high_cutoff=40.0,
    order=4
):

    nyquist_frequency = (
        sampling_rate / 2
    )

    low_normalized = (
        low_cutoff
        / nyquist_frequency
    )

    high_normalized = (
        high_cutoff
        / nyquist_frequency
    )

    sos = butter(
        order,
        [
            low_normalized,
            high_normalized
        ],
        btype="bandpass",
        output="sos"
    )

    filtered_signal = sosfiltfilt(
        sos,
        signal
    )

    return filtered_signal
def build_features(
    waveform_segments,
    previous_rr,
    next_rr,
    rr_ratio
):

    baseline = np.median(
        waveform_segments,
        axis=1
    )

    baseline_corrected = (
        waveform_segments
        - baseline[:, np.newaxis]
    )

    max_voltage = np.max(
        baseline_corrected,
        axis=1
    )

    min_voltage = np.min(
        baseline_corrected,
        axis=1
    )

    voltage_range = np.ptp(
        baseline_corrected,
        axis=1
    )

    rms_amplitude = np.sqrt(
        np.mean(
            baseline_corrected ** 2,
            axis=1
        )
    )

    dt = 1 / sampling_frequency

    waveform_area = (
        np.sum(
            np.abs(
                baseline_corrected
            ),
            axis=1
        )
        * dt
    )

    return np.column_stack([
        max_voltage,
        min_voltage,
        voltage_range,
        rms_amplitude,
        waveform_area,
        previous_rr,
        next_rr,
        rr_ratio
    ])
# -----------------------------------
# Recreate the same record-wise split
# -----------------------------------

best_split = None

for seed in range(100):

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=seed
    )

    train_idx, test_idx = next(
        splitter.split(
            segments,
            labels,
            groups=records
        )
    )

    test_labels_temp = labels[
        test_idx
    ]

    classes, counts = np.unique(
        test_labels_temp,
        return_counts=True
    )

    class_counts = dict(
        zip(
            classes,
            counts
        )
    )

    if (
        class_counts.get("F", 0) >= 100
        and class_counts.get("S", 0) >= 300
        and class_counts.get("V", 0) >= 500
    ):
        best_split = (
            seed,
            train_idx,
            test_idx
        )
        break


seed, train_indices, test_indices = (
    best_split
)

print(
    "\nRobustness split seed:",
    seed
)
# -----------------------------------
# Keep primary four heartbeat classes
# -----------------------------------

train_keep = (
    labels[train_indices] != "Q"
)

test_keep = (
    labels[test_indices] != "Q"
)

train_indices = (
    train_indices[
        train_keep
    ]
)

test_indices = (
    test_indices[
        test_keep
    ]
)


train_segments = (
    segments[
        train_indices
    ]
)

test_segments = (
    segments[
        test_indices
    ]
)

y_train = (
    labels[
        train_indices
    ]
)

y_test = (
    labels[
        test_indices
    ]
)

train_records = (
    records[
        train_indices
    ]
)

test_records = (
    records[
        test_indices
    ]
)

train_samples = (
    samples[
        train_indices
    ]
)

test_samples = (
    samples[
        test_indices
    ]
)
train_previous_rr = full_previous_rr[
    train_indices
]

train_next_rr = full_next_rr[
    train_indices
]

train_rr_ratio = full_rr_ratio[
    train_indices
]


test_previous_rr = full_previous_rr[
    test_indices
]

test_next_rr = full_next_rr[
    test_indices
]

test_rr_ratio = full_rr_ratio[
    test_indices
]
# -----------------------------------
# Train Random Forest on clean ECG
# -----------------------------------

X_train_clean = build_features(
    train_segments,
    train_previous_rr,
    train_next_rr,
    train_rr_ratio
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print(
    "\nTraining robustness model..."
)

rf_model.fit(
    X_train_clean,
    y_train
)

print(
    "Training complete!"
)
# -----------------------------------
# Repeated random-noise robustness
# experiment
# -----------------------------------

noise_levels = [
    0.00,
    0.05,
    0.10,
    0.20,
    0.30
]

number_of_repetitions = 10

robustness_results = []


print(
    "\nREPEATED RANDOM NOISE ROBUSTNESS"
)

print(
    "Repetitions per noise level:",
    number_of_repetitions
)


for noise_std in noise_levels:

    level_accuracies = []
    level_balanced_accuracies = []
    level_macro_f1 = []

    for repetition in range(
        number_of_repetitions
    ):

        if noise_std == 0:

            noisy_test_segments = (
                test_segments.copy()
            )

        else:

            rng = np.random.default_rng(
                repetition
            )

            added_noise = rng.normal(
                loc=0.0,
                scale=noise_std,
                size=test_segments.shape
            )

            noisy_test_segments = (
                test_segments
                + added_noise
            )


        X_test_noisy = build_features(
            noisy_test_segments,
            test_previous_rr,
            test_next_rr,
            test_rr_ratio
        )

        predictions = rf_model.predict(
            X_test_noisy
        )


        level_accuracies.append(
            accuracy_score(
                y_test,
                predictions
            )
        )

        level_balanced_accuracies.append(
            balanced_accuracy_score(
                y_test,
                predictions
            )
        )

        level_macro_f1.append(
            f1_score(
                y_test,
                predictions,
                average="macro"
            )
        )


    mean_accuracy = np.mean(
        level_accuracies
    )

    std_accuracy = np.std(
        level_accuracies
    )

    mean_balanced_accuracy = np.mean(
        level_balanced_accuracies
    )

    std_balanced_accuracy = np.std(
        level_balanced_accuracies
    )

    mean_macro_f1 = np.mean(
        level_macro_f1
    )

    std_macro_f1 = np.std(
        level_macro_f1
    )


    robustness_results.append({
        "noise": noise_std,
        "accuracy_mean": mean_accuracy,
        "accuracy_std": std_accuracy,
        "balanced_mean": mean_balanced_accuracy,
        "balanced_std": std_balanced_accuracy,
        "f1_mean": mean_macro_f1,
        "f1_std": std_macro_f1
    })


    print(
        "\nNoise SD:",
        noise_std,
        "mV"
    )

    print(
        "Accuracy:",
        round(
            mean_accuracy * 100,
            2
        ),
        "+/-",
        round(
            std_accuracy * 100,
            2
        ),
        "%"
    )

    print(
        "Balanced accuracy:",
        round(
            mean_balanced_accuracy * 100,
            2
        ),
        "+/-",
        round(
            std_balanced_accuracy * 100,
            2
        ),
        "%"
    )

    print(
        "Macro F1:",
        round(
            mean_macro_f1,
            3
        ),
        "+/-",
        round(
            std_macro_f1,
            3
        )
    )
    # -----------------------------------
# Per-class noise sensitivity
# -----------------------------------

from sklearn.metrics import recall_score

heartbeat_classes = [
    "N",
    "S",
    "V",
    "F"
]

class_recall_results = {
    heartbeat_class: []
    for heartbeat_class
    in heartbeat_classes
}


print(
    "\nPER-CLASS NOISE SENSITIVITY"
)


for noise_std in noise_levels:

    repetition_recalls = {
        heartbeat_class: []
        for heartbeat_class
        in heartbeat_classes
    }

    for repetition in range(
        number_of_repetitions
    ):

        if noise_std == 0:

            noisy_test_segments = (
                test_segments.copy()
            )

        else:

            rng = np.random.default_rng(
                repetition
            )

            added_noise = rng.normal(
                loc=0.0,
                scale=noise_std,
                size=test_segments.shape
            )

            noisy_test_segments = (
                test_segments
                + added_noise
            )


        X_test_noisy = build_features(
            noisy_test_segments,
            test_previous_rr,
            test_next_rr,
            test_rr_ratio
        )

        predictions = rf_model.predict(
            X_test_noisy
        )


        recalls = recall_score(
            y_test,
            predictions,
            labels=heartbeat_classes,
            average=None,
            zero_division=0
        )


        for heartbeat_class, recall in zip(
            heartbeat_classes,
            recalls
        ):

            repetition_recalls[
                heartbeat_class
            ].append(
                recall
            )


    print(
        "\nNoise SD:",
        noise_std,
        "mV"
    )


    for heartbeat_class in heartbeat_classes:

        mean_recall = np.mean(
            repetition_recalls[
                heartbeat_class
            ]
        )

        std_recall = np.std(
            repetition_recalls[
                heartbeat_class
            ]
        )


        class_recall_results[
            heartbeat_class
        ].append(
            mean_recall
        )


        print(
            heartbeat_class,
            "recall:",
            round(
                mean_recall * 100,
                2
            ),
            "+/-",
            round(
                std_recall * 100,
                2
            ),
            "%"
        )
        # -----------------------------------
# Final noise robustness figure
# -----------------------------------

noise_values = np.array(
    noise_levels
)

accuracy_means = np.array([
    result["accuracy_mean"] * 100
    for result in robustness_results
])

accuracy_stds = np.array([
    result["accuracy_std"] * 100
    for result in robustness_results
])

balanced_means = np.array([
    result["balanced_mean"] * 100
    for result in robustness_results
])

balanced_stds = np.array([
    result["balanced_std"] * 100
    for result in robustness_results
])

f1_means = np.array([
    result["f1_mean"] * 100
    for result in robustness_results
])

f1_stds = np.array([
    result["f1_std"] * 100
    for result in robustness_results
])


plt.figure(
    figsize=(9, 6)
)

plt.errorbar(
    noise_values,
    accuracy_means,
    yerr=accuracy_stds,
    marker="o",
    capsize=4,
    label="Accuracy"
)

plt.errorbar(
    noise_values,
    balanced_means,
    yerr=balanced_stds,
    marker="o",
    capsize=4,
    label="Balanced Accuracy"
)

plt.errorbar(
    noise_values,
    f1_means,
    yerr=f1_stds,
    marker="o",
    capsize=4,
    label="Macro F1"
)

plt.xlabel(
    "Added Gaussian Noise SD (mV)"
)

plt.ylabel(
    "Performance (%)"
)

plt.title(
    "Random Forest ECG Classification Robustness to Added Noise"
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/random_noise_robustness_curve.png",
    dpi=300
)

plt.close()
# -----------------------------------
# Per-class recall robustness figure
# -----------------------------------

plt.figure(
    figsize=(9, 6)
)

for heartbeat_class in heartbeat_classes:

    recalls = (
        np.array(
            class_recall_results[
                heartbeat_class
            ]
        )
        * 100
    )

    plt.plot(
        noise_values,
        recalls,
        marker="o",
        label=heartbeat_class
    )


plt.xlabel(
    "Added Gaussian Noise SD (mV)"
)

plt.ylabel(
    "Recall (%)"
)

plt.title(
    "Heartbeat-Class Sensitivity to ECG Noise"
)

plt.grid(
    alpha=0.3
)

plt.legend(
    title="Heartbeat Class"
)

plt.tight_layout()

plt.savefig(
    "results/per_class_noise_sensitivity.png",
    dpi=300
)

plt.close()
# -----------------------------------
# Filtering recovery demonstration
# -----------------------------------

filter_test_noise_std = 0.20

rng = np.random.default_rng(42)

filter_test_noise = rng.normal(
    loc=0.0,
    scale=filter_test_noise_std,
    size=len(clean_beat)
)

filter_test_corrupted = (
    clean_beat
    + filter_test_noise
)

filter_test_filtered = (
    bandpass_filter_ecg(
        filter_test_corrupted,
        sampling_rate=sampling_frequency
    )
)


corrupted_rmse = np.sqrt(
    np.mean(
        (
            filter_test_corrupted
            - clean_beat
        ) ** 2
    )
)

filtered_rmse = np.sqrt(
    np.mean(
        (
            filter_test_filtered
            - clean_beat
        ) ** 2
    )
)


print(
    "\nFILTERING RECOVERY TEST"
)

print(
    "Noise SD:",
    filter_test_noise_std,
    "mV"
)

print(
    "Corrupted RMSE:",
    round(
        corrupted_rmse,
        4
    ),
    "mV"
)

print(
    "Filtered RMSE:",
    round(
        filtered_rmse,
        4
    ),
    "mV"
)
plt.figure(
    figsize=(10, 6)
)

plt.plot(
    time,
    clean_beat,
    label="Original ECG"
)

plt.plot(
    time,
    filter_test_corrupted,
    label="Noisy ECG",
    alpha=0.7
)

plt.plot(
    time,
    filter_test_filtered,
    label="Filtered ECG"
)

plt.xlabel(
    "Time Relative to Beat (s)"
)

plt.ylabel(
    "ECG Voltage (mV)"
)

plt.title(
    "ECG Recovery After Bandpass Filtering"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "results/filtering_recovery_example.png",
    dpi=300
)

plt.close()
# -----------------------------------
# Baseline-corrected filtering comparison
# -----------------------------------

clean_beat_corrected = (
    clean_beat
    - np.median(clean_beat)
)

corrupted_beat_corrected = (
    filter_test_corrupted
    - np.median(filter_test_corrupted)
)

filtered_beat_corrected = (
    filter_test_filtered
    - np.median(filter_test_filtered)
)


corrupted_corrected_rmse = np.sqrt(
    np.mean(
        (
            corrupted_beat_corrected
            - clean_beat_corrected
        ) ** 2
    )
)

filtered_corrected_rmse = np.sqrt(
    np.mean(
        (
            filtered_beat_corrected
            - clean_beat_corrected
        ) ** 2
    )
)


print(
    "\nBASELINE-CORRECTED FILTER TEST"
)

print(
    "Noisy RMSE:",
    round(
        corrupted_corrected_rmse,
        4
    ),
    "mV"
)

print(
    "Filtered RMSE:",
    round(
        filtered_corrected_rmse,
        4
    ),
    "mV"
)
plt.figure(
    figsize=(10, 6)
)

plt.plot(
    time,
    clean_beat_corrected,
    label="Original ECG"
)

plt.plot(
    time,
    corrupted_beat_corrected,
    label="Noisy ECG",
    alpha=0.7
)

plt.plot(
    time,
    filtered_beat_corrected,
    label="Filtered ECG"
)

plt.xlabel(
    "Time Relative to Beat (s)"
)

plt.ylabel(
    "Baseline-Corrected ECG Voltage (mV)"
)

plt.title(
    "Baseline-Corrected ECG Filtering Comparison"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "results/baseline_corrected_filtering_example.png",
    dpi=300
)

plt.close()
# -----------------------------------
# Full-dataset filtering recovery test
# -----------------------------------

filter_noise_std = 0.20
filter_repetitions = 10

unfiltered_accuracy = []
filtered_accuracy = []

unfiltered_balanced = []
filtered_balanced = []

unfiltered_f1 = []
filtered_f1 = []
unfiltered_class_recalls = {
    heartbeat_class: []
    for heartbeat_class
    in heartbeat_classes
}

filtered_class_recalls = {
    heartbeat_class: []
    for heartbeat_class
    in heartbeat_classes
}


print(
    "\nFULL-DATASET FILTERING RECOVERY"
)


for repetition in range(
    filter_repetitions
):

    rng = np.random.default_rng(
        repetition
    )

    added_noise = rng.normal(
        loc=0.0,
        scale=filter_noise_std,
        size=test_segments.shape
    )

    noisy_segments = (
        test_segments
        + added_noise
    )


    # Filter every heartbeat segment
    filtered_segments = np.array([
        bandpass_filter_ecg(
            segment,
            sampling_rate=sampling_frequency
        )
        for segment in noisy_segments
    ])


    X_noisy = build_features(
        noisy_segments,
        test_previous_rr,
        test_next_rr,
        test_rr_ratio
    )

    X_filtered = build_features(
        filtered_segments,
        test_previous_rr,
        test_next_rr,
        test_rr_ratio
    )


    noisy_predictions = (
        rf_model.predict(
            X_noisy
        )
    )

    filtered_predictions = (
        rf_model.predict(
            X_filtered
        )
    )
    noisy_recalls = recall_score(
    y_test,
    noisy_predictions,
    labels=heartbeat_classes,
    average=None,
    zero_division=0
)

filtered_recalls = recall_score(
    y_test,
    filtered_predictions,
    labels=heartbeat_classes,
    average=None,
    zero_division=0
)


for heartbeat_class, recall in zip(
    heartbeat_classes,
    noisy_recalls
):

    unfiltered_class_recalls[
        heartbeat_class
    ].append(
        recall
    )


for heartbeat_class, recall in zip(
    heartbeat_classes,
    filtered_recalls
):

    filtered_class_recalls[
        heartbeat_class
    ].append(
        recall
    )


    unfiltered_accuracy.append(
        accuracy_score(
            y_test,
            noisy_predictions
        )
    )

    filtered_accuracy.append(
        accuracy_score(
            y_test,
            filtered_predictions
        )
    )


    unfiltered_balanced.append(
        balanced_accuracy_score(
            y_test,
            noisy_predictions
        )
    )

    filtered_balanced.append(
        balanced_accuracy_score(
            y_test,
            filtered_predictions
        )
    )


    unfiltered_f1.append(
        f1_score(
            y_test,
            noisy_predictions,
            average="macro"
        )
    )

    filtered_f1.append(
        f1_score(
            y_test,
            filtered_predictions,
            average="macro"
        )
    )


print(
    "Noise SD:",
    filter_noise_std,
    "mV"
)

print(
    "\nUNFILTERED NOISY ECG"
)

print(
    "Accuracy:",
    round(
        np.mean(
            unfiltered_accuracy
        ) * 100,
        2
    ),
    "%"
)

print(
    "Balanced accuracy:",
    round(
        np.mean(
            unfiltered_balanced
        ) * 100,
        2
    ),
    "%"
)

print(
    "Macro F1:",
    round(
        np.mean(
            unfiltered_f1
        ),
        3
    )
)


print(
    "\nFILTERED ECG"
)

print(
    "Accuracy:",
    round(
        np.mean(
            filtered_accuracy
        ) * 100,
        2
    ),
    "%"
)

print(
    "Balanced accuracy:",
    round(
        np.mean(
            filtered_balanced
        ) * 100,
        2
    ),
    "%"
)

print(
    "Macro F1:",
    round(
        np.mean(
            filtered_f1
        ),
        3
    )
)
print(
    "\nPER-CLASS FILTERING RECOVERY"
)

for heartbeat_class in heartbeat_classes:

    noisy_mean = np.mean(
        unfiltered_class_recalls[
            heartbeat_class
        ]
    )

    filtered_mean = np.mean(
        filtered_class_recalls[
            heartbeat_class
        ]
    )

    print(
        "\nClass:",
        heartbeat_class
    )

    print(
        "Unfiltered recall:",
        round(
            noisy_mean * 100,
            2
        ),
        "%"
    )

    print(
        "Filtered recall:",
        round(
            filtered_mean * 100,
            2
        ),
        "%"
    )
    # -----------------------------------
# Baseline-wander robustness experiment
# -----------------------------------

baseline_amplitudes = [
    0.00,
    0.05,
    0.10,
    0.20,
    0.30
]

baseline_frequency = 0.5
baseline_repetitions = 10

baseline_results = []

segment_time = (
    np.arange(test_segments.shape[1])
    / sampling_frequency
)


print(
    "\nBASELINE-WANDER ROBUSTNESS"
)


for amplitude in baseline_amplitudes:

    level_accuracy = []
    level_balanced = []
    level_f1 = []

    for repetition in range(
        baseline_repetitions
    ):

        if amplitude == 0:

            baseline_corrupted_segments = (
                test_segments.copy()
            )

        else:

            rng = np.random.default_rng(
                repetition
            )

            phases = rng.uniform(
                0,
                2 * np.pi,
                size=test_segments.shape[0]
            )

            baseline_wander_matrix = (
                amplitude
                * np.sin(
                    2
                    * np.pi
                    * baseline_frequency
                    * segment_time[np.newaxis, :]
                    + phases[:, np.newaxis]
                )
            )

            baseline_corrupted_segments = (
                test_segments
                + baseline_wander_matrix
            )


        X_baseline = build_features(
            baseline_corrupted_segments,
            test_previous_rr,
            test_next_rr,
            test_rr_ratio
        )

        predictions = rf_model.predict(
            X_baseline
        )


        level_accuracy.append(
            accuracy_score(
                y_test,
                predictions
            )
        )

        level_balanced.append(
            balanced_accuracy_score(
                y_test,
                predictions
            )
        )

        level_f1.append(
            f1_score(
                y_test,
                predictions,
                average="macro"
            )
        )


    mean_accuracy = np.mean(
        level_accuracy
    )

    std_accuracy = np.std(
        level_accuracy
    )

    mean_balanced = np.mean(
        level_balanced
    )

    std_balanced = np.std(
        level_balanced
    )

    mean_f1 = np.mean(
        level_f1
    )

    std_f1 = np.std(
        level_f1
    )


    baseline_results.append({
        "amplitude": amplitude,
        "accuracy_mean": mean_accuracy,
        "accuracy_std": std_accuracy,
        "balanced_mean": mean_balanced,
        "balanced_std": std_balanced,
        "f1_mean": mean_f1,
        "f1_std": std_f1
    })


    print(
        "\nBaseline amplitude:",
        amplitude,
        "mV"
    )

    print(
        "Accuracy:",
        round(
            mean_accuracy * 100,
            2
        ),
        "+/-",
        round(
            std_accuracy * 100,
            2
        ),
        "%"
    )

    print(
        "Balanced accuracy:",
        round(
            mean_balanced * 100,
            2
        ),
        "+/-",
        round(
            std_balanced * 100,
            2
        ),
        "%"
    )

    print(
        "Macro F1:",
        round(
            mean_f1,
            3
        ),
        "+/-",
        round(
            std_f1,
            3
        )
    )