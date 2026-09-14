import streamlit as st
import wfdb
import numpy as np
import plotly.graph_objects as go
import neurokit2 as nk
import joblib
# -----------------------------------
# Load trained Random Forest model
# -----------------------------------

rf_model = joblib.load(
    "results/random_forest_model.joblib"
)
def build_single_beat_features(
    beat_segment,
    previous_rr,
    next_rr
):

    sampling_frequency = 360

    baseline = np.median(
        beat_segment
    )

    baseline_corrected = (
        beat_segment
        - baseline
    )

    max_voltage = np.max(
        baseline_corrected
    )

    min_voltage = np.min(
        baseline_corrected
    )

    voltage_range = np.ptp(
        baseline_corrected
    )

    rms_amplitude = np.sqrt(
        np.mean(
            baseline_corrected ** 2
        )
    )

    dt = 1 / sampling_frequency

    waveform_area = (
        np.sum(
            np.abs(
                baseline_corrected
            )
        )
        * dt
    )

    local_rr_average = (
        previous_rr
        + next_rr
    ) / 2

    if local_rr_average > 0:

        rr_ratio = (
            previous_rr
            / local_rr_average
        )

    else:

        rr_ratio = 1.0


    features = np.array([[
        max_voltage,
        min_voltage,
        voltage_range,
        rms_amplitude,
        waveform_area,
        previous_rr,
        next_rr,
        rr_ratio
    ]])

    feature_values = {
        "Maximum Voltage": max_voltage,
        "Minimum Voltage": min_voltage,
        "Voltage Range": voltage_range,
        "RMS Amplitude": rms_amplitude,
        "Waveform Area": waveform_area,
        "Previous RR": previous_rr,
        "Next RR": next_rr,
        "RR Ratio": rr_ratio
    }

    return features, feature_values
st.set_page_config(
    page_title="ECG Arrhythmia Analysis",
    page_icon="🫀",
    layout="wide"
)

st.title(
    "ECG Arrhythmia Analysis System"
)
st.caption(
    """
    Biomedical signal processing, physiological feature engineering,
    machine-learning classification, and robustness analysis using
    the MIT-BIH Arrhythmia Database.
    """
)

st.write(
    """
    Biomedical engineering project for ECG signal processing,
    heartbeat analysis, and machine-learning classification.
    """
)

st.info(
    """
    Educational engineering prototype only.
    This application is not a clinical diagnostic device.
    """
)
st.warning(
    """
    **Demonstration vs. validation:** The interactive waveform
    below uses MIT-BIH Record 100 as an illustrative example.
    Record 100 was used during model development/training, so
    individual predictions shown in this interactive section
    should not be interpreted as held-out validation results.

    Model generalization was evaluated separately using
    record-wise held-out testing and grouped cross-validation.
    """
)
# -----------------------------------
# Load MIT-BIH ECG
# -----------------------------------

st.header(
    "ECG Signal Explorer"
)

record = wfdb.rdrecord(
    "data/100"
)
annotation = wfdb.rdann(
    "data/100",
    "atr"
)

sampling_frequency = record.fs

ecg_signal = record.p_signal[:, 0]


# Display first 10 seconds
duration = 10

number_of_samples = int(
    duration * sampling_frequency
)

ecg_window = ecg_signal[
    :number_of_samples
]

time = (
    np.arange(number_of_samples)
    / sampling_frequency
)
# -----------------------------------
# ECG cleaning and R-peak detection
# -----------------------------------

cleaned_ecg = nk.ecg_clean(
    ecg_window,
    sampling_rate=sampling_frequency,
    method="neurokit"
)

signals, info = nk.ecg_peaks(
    cleaned_ecg,
    sampling_rate=sampling_frequency
)

r_peaks = info[
    "ECG_R_Peaks"
]

r_peak_times = (
    r_peaks
    / sampling_frequency
)

rr_intervals = (
    np.diff(r_peaks)
    / sampling_frequency
)

heart_rates = (
    60
    / rr_intervals
)

average_heart_rate = np.mean(
    heart_rates
)
# -----------------------------------
# Expert heartbeat annotations
# -----------------------------------

beat_symbols = [
    "N", "L", "R", "A", "a",
    "J", "S", "V", "F",
    "e", "j", "E", "/"
]

expert_samples = []
expert_symbols = []

for sample, symbol in zip(
    annotation.sample,
    annotation.symbol
):

    if sample >= number_of_samples:
        break

    if symbol in beat_symbols:

        expert_samples.append(
            sample
        )

        expert_symbols.append(
            symbol
        )

expert_samples = np.array(
    expert_samples
)

expert_times = (
    expert_samples
    / sampling_frequency
)


st.write(
    "MIT-BIH Record 100 — first 10 seconds"
)


# -----------------------------------
# Interactive ECG plot
# -----------------------------------

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=time,
        y=ecg_window,
        mode="lines",
        name="ECG"
    )
)
fig.add_trace(
    go.Scatter(
        x=r_peak_times,
        y=ecg_window[r_peaks],
        mode="markers",
        name="Detected R-Peaks",
        marker=dict(
            size=9,
            symbol="circle"
        )
    )
)
fig.add_trace(
    go.Scatter(
        x=expert_times,
        y=ecg_window[
            expert_samples
        ],
        mode="markers+text",
        name="Expert Beat Labels",
        text=expert_symbols,
        textposition="top center",
        marker=dict(
            size=7,
            symbol="diamond"
        )
    )
)

fig.update_layout(
    title="Raw ECG Waveform",
    xaxis_title="Time (seconds)",
    yaxis_title="ECG Voltage (mV)",
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# -----------------------------------
# Recording information
# -----------------------------------

col1, col2, col3 = st.columns(3)

col1.metric(
    "Sampling Rate",
    f"{sampling_frequency:.0f} Hz"
)

col2.metric(
    "Displayed Duration",
    f"{duration} s"
)

col3.metric(
    "ECG Samples",
    f"{number_of_samples:,}"
)
st.subheader(
    "Cardiac Timing Analysis"
)

heart_col1, heart_col2, heart_col3 = (
    st.columns(3)
)

heart_col1.metric(
    "Detected Heartbeats",
    len(r_peaks)
)

heart_col2.metric(
    "Average Heart Rate",
    f"{average_heart_rate:.1f} BPM"
)

heart_col3.metric(
    "Average RR Interval",
    f"{np.mean(rr_intervals):.3f} s"
)
st.subheader(
    "Expert-Labeled Heartbeats"
)

st.write(
    f"""
    This 10-second ECG contains
    **{len(expert_samples)} expert-annotated heartbeats**.
    The letters above the ECG come from the
    MIT-BIH expert annotations.
    """
)

st.write(
    """
    **N** = Normal beat  
    **A** = Atrial premature beat
    """
)

if "A" in expert_symbols:

    atrial_index = (
        expert_symbols.index("A")
    )

    atrial_time = (
        expert_times[
            atrial_index
        ]
    )

    st.warning(
        f"""
        An atrial premature beat occurs at approximately
        **{atrial_time:.2f} seconds**.

        Notice that this beat occurs earlier than the
        surrounding rhythm. This produces a shortened
        preceding RR interval and demonstrates why
        average heart rate alone can miss important
        beat-to-beat abnormalities.
        """
    )# -----------------------------------
# Machine-learning heartbeat analysis
# -----------------------------------
# -----------------------------------
# Interactive heartbeat selector
# -----------------------------------

st.subheader(
    "Interactive Heartbeat Selection"
)

heartbeat_options = []

for index, (
    symbol,
    beat_time
) in enumerate(
    zip(
        expert_symbols,
        expert_times
    )
):

    heartbeat_options.append(
        f"Beat {index + 1}: "
        f"{symbol} at "
        f"{beat_time:.3f} s"
    )


selected_heartbeat = st.selectbox(
    "Select a heartbeat to analyze:",
    heartbeat_options,
    index=7
)

selected_index = (
    heartbeat_options.index(
        selected_heartbeat
    )
)

selected_symbol = (
    expert_symbols[
        selected_index
    ]
)

selected_sample = (
    expert_samples[
        selected_index
    ]
)

selected_time = (
    expert_times[
        selected_index
    ]
)


st.write(
    f"""
    **Selected beat:** {selected_symbol}  
    **Time:** {selected_time:.3f} seconds
    """
)
st.subheader(
    "Machine-Learning Beat Classification"
)

# -----------------------------------
# Analyze selected heartbeat
# -----------------------------------

selected_global_index = np.where(
    annotation.sample == selected_sample
)[0][0]


# -----------------------------------
# RR timing
# -----------------------------------

if selected_global_index > 0:

    previous_sample = (
        annotation.sample[
            selected_global_index - 1
        ]
    )

    previous_rr = (
        selected_sample
        - previous_sample
    ) / sampling_frequency

else:

    previous_rr = 0.0


if selected_global_index < (
    len(annotation.sample) - 1
):

    next_sample = (
        annotation.sample[
            selected_global_index + 1
        ]
    )

    next_rr = (
        next_sample
        - selected_sample
    ) / sampling_frequency

else:

    next_rr = 0.0


# -----------------------------------
# Extract beat-centered ECG segment
# -----------------------------------

before_samples = int(
    0.3
    * sampling_frequency
)

after_samples = int(
    0.4
    * sampling_frequency
)

segment_start = (
    selected_sample
    - before_samples
)

segment_end = (
    selected_sample
    + after_samples
)


if (
    segment_start >= 0
    and segment_end
    <= len(ecg_signal)
):

    beat_segment = ecg_signal[
        segment_start:
        segment_end
    ]
        # -----------------------------------
    # Selected heartbeat waveform
    # -----------------------------------

    beat_time = (
        np.arange(
            len(beat_segment)
        )
        / sampling_frequency
        - 0.3
    )

    beat_fig = go.Figure()

    beat_fig.add_trace(
        go.Scatter(
            x=beat_time,
            y=beat_segment,
            mode="lines",
            name="Selected Heartbeat"
        )
    )

    beat_fig.add_vline(
        x=0,
        line_dash="dash",
        annotation_text="Expert Beat Location"
    )

    beat_fig.update_layout(
        title=(
            f"Selected Heartbeat: "
            f"{selected_symbol} at "
            f"{selected_time:.3f} s"
        ),
        xaxis_title=(
            "Time Relative to Beat (seconds)"
        ),
        yaxis_title="ECG Voltage (mV)"
    )

    st.plotly_chart(
        beat_fig,
        use_container_width=True
    )
    


    beat_features, feature_values = (
        build_single_beat_features(
            beat_segment,
            previous_rr,
            next_rr
        )
    )

        # -----------------------------------
    # Physiological feature display
    # -----------------------------------

    st.subheader(
        "Features Used by the Classifier"
    )

    feature_col1, feature_col2, feature_col3, feature_col4 = (
        st.columns(4)
    )

    feature_col1.metric(
        "Voltage Range",
        f"{feature_values['Voltage Range']:.3f} mV"
    )

    feature_col2.metric(
        "RMS Amplitude",
        f"{feature_values['RMS Amplitude']:.3f} mV"
    )

    feature_col3.metric(
        "Waveform Area",
        f"{feature_values['Waveform Area']:.4f} mV·s"
    )

    feature_col4.metric(
        "RR Ratio",
        f"{feature_values['RR Ratio']:.3f}"
    )


    timing_col1, timing_col2 = st.columns(2)

    timing_col1.metric(
        "Previous RR",
        f"{feature_values['Previous RR']:.3f} s"
    )

    timing_col2.metric(
        "Next RR",
        f"{feature_values['Next RR']:.3f} s"
    )


    with st.expander(
        "Show all 8 model features"
    ):

        st.write(
            f"""
            **Maximum voltage:** {feature_values['Maximum Voltage']:.3f} mV  
            **Minimum voltage:** {feature_values['Minimum Voltage']:.3f} mV  
            **Voltage range:** {feature_values['Voltage Range']:.3f} mV  
            **RMS amplitude:** {feature_values['RMS Amplitude']:.3f} mV  
            **Waveform area:** {feature_values['Waveform Area']:.4f} mV·s  
            **Previous RR:** {feature_values['Previous RR']:.3f} s  
            **Next RR:** {feature_values['Next RR']:.3f} s  
            **RR ratio:** {feature_values['RR Ratio']:.3f}
            """
        )


    predicted_class = (
        rf_model.predict(
            beat_features
        )[0]
    )


    prediction_probabilities = (
        rf_model.predict_proba(
            beat_features
        )[0]
    )

    class_labels = (
        rf_model.classes_
    )

    confidence = np.max(
        prediction_probabilities
    )


    # -----------------------------------
    # Map raw expert label
    # to grouped class
    # -----------------------------------

    if selected_symbol in [
        "N", "L", "R", "e", "j"
    ]:

        grouped_expert_class = "N"

    elif selected_symbol in [
        "A", "a", "J", "S"
    ]:

        grouped_expert_class = "S"

    elif selected_symbol in [
        "V", "E"
    ]:

        grouped_expert_class = "V"

    elif selected_symbol == "F":

        grouped_expert_class = "F"

    else:

        grouped_expert_class = "Q"


    # -----------------------------------
    # Display classification result
    # -----------------------------------

    result_col1, result_col2, result_col3 = (
        st.columns(3)
    )

    result_col1.metric(
        "Expert Raw Label",
        selected_symbol
    )

    result_col2.metric(
        "Grouped Expert Class",
        grouped_expert_class
    )

    result_col3.metric(
        "Model Prediction",
        predicted_class
    )


    st.write(
        f"""
        **Previous RR:** {previous_rr:.3f} s  
        **Next RR:** {next_rr:.3f} s  
        **Model probability estimate:** {confidence * 100:.1f}%
        """
    )


    if (
        predicted_class
        == grouped_expert_class
    ):

        st.success(
            "Model prediction matches the grouped expert label."
        )

    else:

        st.error(
            "Model prediction does not match the grouped expert label."
        )


    # -----------------------------------
    # Probability chart
    # -----------------------------------

    st.write(
        "### Model Probability Estimates"
    )

    probability_fig = go.Figure()

    probability_fig.add_trace(
        go.Bar(
            x=class_labels,
            y=prediction_probabilities
            * 100
        )
    )

    probability_fig.update_layout(
        xaxis_title="Heartbeat Class",
        yaxis_title=(
            "Model Probability Estimate (%)"
        ),
        yaxis_range=[0, 100],
        title=(
            "Random Forest Prediction "
            "Distribution"
        )
    )

    st.plotly_chart(
        probability_fig,
        use_container_width=True
    )

    st.caption(
        """
        These values are Random Forest
        probability estimates, not calibrated
        clinical probabilities or diagnostic
        certainty.
        """
    )

else:

    st.warning(
        """
        This beat is too close to the recording
        boundary to create the full 0.7-second
        analysis window.
        """
    )

    # -----------------------------------
# Signal quality and robustness
# -----------------------------------

st.divider()

st.header(
    "Signal Quality & Robustness"
)

st.write(
    """
    ECG measurements can be affected by movement,
    muscle activity, electrode contact, and other
    sources of noise. A biomedical engineering
    system should therefore evaluate not only
    classification performance, but also how
    performance changes when signal quality
    decreases.
    """
)

st.subheader(
    "Controlled Noise Experiment"
)

st.write(
    """
    Synthetic Gaussian noise was added to held-out
    ECG beats at increasing amplitudes. The same
    trained Random Forest was then evaluated without
    retraining.
    """
)


noise_levels = [
    0.00,
    0.05,
    0.10,
    0.20,
    0.30
]

noise_accuracy = [
    88.08,
    87.94,
    86.89,
    84.27,
    78.26
]

noise_macro_f1 = [
    0.479,
    0.483,
    0.466,
    0.388,
    0.375
]


robustness_fig = go.Figure()

robustness_fig.add_trace(
    go.Scatter(
        x=noise_levels,
        y=noise_accuracy,
        mode="lines+markers",
        name="Accuracy (%)"
    )
)

robustness_fig.update_layout(
    title=(
        "Classification Performance "
        "Under Gaussian Noise"
    ),
    xaxis_title=(
        "Added Noise Standard Deviation (mV)"
    ),
    yaxis_title="Accuracy (%)"
)

st.plotly_chart(
    robustness_fig,
    use_container_width=True
)


st.write(
    """
    As noise increased from **0.00 mV to 0.30 mV**,
    classification accuracy decreased from
    **88.08% to 78.26%**.

    This demonstrates that reliable heartbeat
    detection and reliable heartbeat classification
    are separate engineering problems.
    """
)
st.subheader(
    "Can Filtering Recover Performance?"
)

filter_col1, filter_col2 = st.columns(2)

filter_col1.metric(
    "Noisy Balanced Accuracy",
    "46.93%"
)

filter_col2.metric(
    "Filtered Balanced Accuracy",
    "51.66%",
    delta="+4.73 percentage points"
)

filter_col3, filter_col4 = st.columns(2)

filter_col3.metric(
    "Noisy S-Class Recall",
    "2.65%"
)

filter_col4.metric(
    "Filtered S-Class Recall",
    "20.56%",
    delta="+17.91 percentage points"
)

st.write(
    """
    At **0.20 mV Gaussian noise**, applying a
    0.5–40 Hz band-pass filter improved balanced
    accuracy and substantially recovered detection
    of supraventricular ectopic beats.

    However, filtering did not improve every metric.
    Overall accuracy decreased slightly from
    **84.27% to 83.65%**. This illustrates an
    important biomedical engineering tradeoff:
    preprocessing can improve detection of some
    clinically relevant minority patterns without
    improving overall accuracy.
    """
)
# -----------------------------------
# Model interpretation
# -----------------------------------

st.divider()

st.header(
    "Model Interpretation"
)

st.write(
    """
    The Random Forest does not use every feature
    equally. Feature importance estimates show
    which measurements were most useful to the
    model when separating heartbeat classes.
    """
)

feature_names = [
    "RR Ratio",
    "Waveform Area",
    "Maximum Voltage",
    "RMS Amplitude",
    "Previous RR",
    "Next RR",
    "Minimum Voltage",
    "Voltage Range"
]

feature_importance = [
    0.1611,
    0.1588,
    0.1326,
    0.1301,
    0.1265,
    0.1153,
    0.1063,
    0.0693
]

importance_fig = go.Figure()

importance_fig.add_trace(
    go.Bar(
        x=feature_importance,
        y=feature_names,
        orientation="h"
    )
)

importance_fig.update_layout(
    title="Random Forest Feature Importance",
    xaxis_title="Importance",
    yaxis_title="Feature"
)

st.plotly_chart(
    importance_fig,
    use_container_width=True
)

st.write(
    """
    The two most influential features were
    **RR ratio** and **waveform area**.

    - **RR ratio** describes how early or late a
      heartbeat occurs relative to surrounding beats.
    - **Waveform area** summarizes the overall
      magnitude of the electrical waveform across
      the heartbeat segment.

    This shows that the classifier is using both
    **timing information** and **ECG morphology**,
    rather than relying on a single type of signal
    measurement.
    """
)

st.caption(
    """
    Feature importance describes how useful a feature
    was to this trained model. It does not prove that
    the feature directly causes an arrhythmia.
    """
)
# -----------------------------------
# Model validation summary
# -----------------------------------

st.divider()

st.header(
    "Model Validation"
)

st.write(
    """
    To reduce information leakage, ECG recordings were separated
    by record rather than randomly splitting individual heartbeats.
    This prevents beats from the same recording from appearing in
    both the training and held-out test sets.
    """
)

validation_col1, validation_col2, validation_col3 = st.columns(3)

validation_col1.metric(
    "Held-Out Accuracy",
    "88.08%"
)

validation_col2.metric(
    "V-Class Recall",
    "97.5%"
)

validation_col3.metric(
    "Balanced Accuracy",
    "53.65%"
)

st.write(
    """
    The Random Forest achieved **88.08% overall accuracy** on
    held-out records and detected **97.5% of ventricular ectopic
    beats (V)**. However, performance was substantially weaker for
    supraventricular (S) and fusion (F) beats.

    This is why overall accuracy alone is not sufficient for
    evaluating an imbalanced biomedical classification problem.
    """
)

st.subheader(
    "Grouped Cross-Validation"
)

st.write(
    """
    Five-fold GroupKFold validation was also performed using
    training records. The Random Forest achieved a mean balanced
    accuracy of **46.92% ± 1.32 percentage points**.

    The difference between high overall accuracy and much lower
    balanced accuracy demonstrates an important limitation of the
    model: performance is considerably stronger for some heartbeat
    classes than others.
    """
)

st.caption(
    """
    These results describe an engineering prototype evaluated on
    the MIT-BIH Arrhythmia Database and should not be interpreted
    as clinical diagnostic performance.
    """
)
# -----------------------------------
# Engineering limitations
# -----------------------------------

st.divider()

st.header(
    "Engineering Limitations"
)

st.write(
    """
    This project demonstrates an ECG analysis pipeline, but several
    limitations prevent it from being interpreted as a clinical
    diagnostic system.
    """
)

st.markdown(
    """
    - Heartbeat segmentation and class labels were based on expert
      MIT-BIH annotations during dataset construction.
    - The primary classifier predicts four grouped classes:
      **N, S, V, and F**; paced/other beats were excluded from the
      main classifier.
    - The model uses **future beat timing (Next RR)**, so it is an
      offline or delayed classifier rather than an instantaneous
      real-time system.
    - ECG recordings were separated by **record**, but this should
      not automatically be interpreted as a guaranteed patient-wise
      split.
    - Performance is affected by class imbalance and differences
      between recordings.
    - Synthetic Gaussian noise and baseline wander are controlled
      engineering stress tests and do not reproduce every real-world
      ECG artifact.
    - Random Forest probability estimates are not calibrated clinical
      probabilities.
    """
)
st.divider()

st.header(
    "Engineering Takeaway"
)

st.success(
    """
    Reliable ECG analysis requires more than a high classification
    accuracy. Signal quality, physiological feature design, class
    imbalance, validation strategy, and error analysis all affect
    whether a biomedical algorithm can be trusted.
    """
)