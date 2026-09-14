import numpy as np
import wfdb
import neurokit2 as nk

# -----------------------------------
# Load continuous ECG
# -----------------------------------

record = wfdb.rdrecord(
    "data/100"
)

annotation = wfdb.rdann(
    "data/100",
    "atr"
)

sampling_frequency = record.fs

ecg_signal = record.p_signal[:, 0]


print(
    "Sampling frequency:",
    sampling_frequency,
    "Hz"
)

print(
    "Total ECG samples:",
    len(ecg_signal)
)
# -----------------------------------
# Clean-signal R-peak detection
# -----------------------------------

analysis_duration = 60  # seconds

analysis_samples = int(
    analysis_duration
    * sampling_frequency
)

ecg_window = ecg_signal[
    :analysis_samples
]


cleaned_ecg = nk.ecg_clean(
    ecg_window,
    sampling_rate=sampling_frequency,
    method="neurokit"
)

signals, info = nk.ecg_peaks(
    cleaned_ecg,
    sampling_rate=sampling_frequency
)

detected_r_peaks = info[
    "ECG_R_Peaks"
]


print(
    "\nCLEAN R-PEAK DETECTION"
)

print(
    "Analysis duration:",
    analysis_duration,
    "seconds"
)

print(
    "Detected R-peaks:",
    len(detected_r_peaks)
)
# -----------------------------------
# Expert heartbeat annotations
# -----------------------------------

beat_symbols = [
    "N", "L", "R", "A", "a",
    "J", "S", "V", "F",
    "e", "j", "E", "/"
]

expert_beats = []

for sample, symbol in zip(
    annotation.sample,
    annotation.symbol
):

    if symbol not in beat_symbols:
        continue

    if sample >= analysis_samples:
        break

    expert_beats.append(
        sample
    )

expert_beats = np.array(
    expert_beats
)


print(
    "Expert heartbeat count:",
    len(expert_beats)
)
# -----------------------------------
# Compare detector with expert beats
# -----------------------------------

tolerance_seconds = 0.05

tolerance_samples = int(
    tolerance_seconds
    * sampling_frequency
)

matched_expert = set()
true_positives = 0
false_positives = 0


for detected_peak in detected_r_peaks:

    distances = np.abs(
        expert_beats
        - detected_peak
    )

    closest_index = np.argmin(
        distances
    )

    if (
        distances[closest_index]
        <= tolerance_samples
        and closest_index
        not in matched_expert
    ):

        true_positives += 1

        matched_expert.add(
            closest_index
        )

    else:

        false_positives += 1


false_negatives = (
    len(expert_beats)
    - true_positives
)


sensitivity = (
    true_positives
    /
    (
        true_positives
        + false_negatives
    )
)

ppv = (
    true_positives
    /
    (
        true_positives
        + false_positives
    )
)


print(
    "\nCLEAN DETECTION PERFORMANCE"
)

print(
    "True positives:",
    true_positives
)

print(
    "False positives:",
    false_positives
)

print(
    "False negatives:",
    false_negatives
)

print(
    "Sensitivity:",
    round(
        sensitivity * 100,
        2
    ),
    "%"
)

print(
    "PPV:",
    round(
        ppv * 100,
        2
    ),
    "%"
)
# -----------------------------------
# R-peak robustness to Gaussian noise
# -----------------------------------

noise_levels = [
    0.00,
    0.05,
    0.10,
    0.20,
    0.30
]

repetitions = 10

print(
    "\nR-PEAK NOISE ROBUSTNESS"
)


for noise_std in noise_levels:

    sensitivities = []
    ppvs = []

    for repetition in range(
        repetitions
    ):

        if noise_std == 0:

            noisy_ecg = (
                ecg_window.copy()
            )

        else:

            rng = np.random.default_rng(
                repetition
            )

            noise = rng.normal(
                loc=0.0,
                scale=noise_std,
                size=len(ecg_window)
            )

            noisy_ecg = (
                ecg_window
                + noise
            )


        cleaned_noisy_ecg = nk.ecg_clean(
            noisy_ecg,
            sampling_rate=sampling_frequency,
            method="neurokit"
        )

        noisy_signals, noisy_info = (
            nk.ecg_peaks(
                cleaned_noisy_ecg,
                sampling_rate=sampling_frequency
            )
        )

        noisy_detected_peaks = (
            noisy_info[
                "ECG_R_Peaks"
            ]
        )


        matched_expert = set()
        true_positives = 0
        false_positives = 0


        for detected_peak in (
            noisy_detected_peaks
        ):

            distances = np.abs(
                expert_beats
                - detected_peak
            )

            closest_index = np.argmin(
                distances
            )

            if (
                distances[
                    closest_index
                ]
                <= tolerance_samples
                and closest_index
                not in matched_expert
            ):

                true_positives += 1

                matched_expert.add(
                    closest_index
                )

            else:

                false_positives += 1


        false_negatives = (
            len(expert_beats)
            - true_positives
        )


        sensitivity = (
            true_positives
            /
            (
                true_positives
                + false_negatives
            )
        )


        if (
            true_positives
            + false_positives
        ) > 0:

            ppv = (
                true_positives
                /
                (
                    true_positives
                    + false_positives
                )
            )

        else:

            ppv = 0


        sensitivities.append(
            sensitivity
        )

        ppvs.append(
            ppv
        )


    print(
        "\nNoise SD:",
        noise_std,
        "mV"
    )

    print(
        "Sensitivity:",
        round(
            np.mean(
                sensitivities
            ) * 100,
            2
        ),
        "+/-",
        round(
            np.std(
                sensitivities
            ) * 100,
            2
        ),
        "%"
    )

    print(
        "PPV:",
        round(
            np.mean(
                ppvs
            ) * 100,
            2
        ),
        "+/-",
        round(
            np.std(
                ppvs
            ) * 100,
            2
        ),
        "%"
    )
    # -----------------------------------
# R-peak robustness figure
# -----------------------------------

noise_values = [
    0.00,
    0.05,
    0.10,
    0.20,
    0.30
]

sensitivity_values = [
    98.65,
    98.65,
    98.65,
    98.65,
    98.38
]

ppv_values = [
    100.00,
    100.00,
    100.00,
    99.86,
    99.32
]

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


plt.figure(
    figsize=(9, 6)
)

plt.plot(
    noise_values,
    sensitivity_values,
    marker="o",
    label="Sensitivity"
)

plt.plot(
    noise_values,
    ppv_values,
    marker="o",
    label="PPV"
)

plt.xlabel(
    "Added Gaussian Noise SD (mV)"
)

plt.ylabel(
    "Detection Performance (%)"
)

plt.title(
    "R-Peak Detection Robustness to Added ECG Noise"
)

plt.ylim(
    95,
    101
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/rpeak_noise_robustness.png",
    dpi=300
)

plt.close()