import numpy as np
import joblib
from sklearn.model_selection import (
    GroupShuffleSplit,
    GroupKFold
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    ConfusionMatrixDisplay
)
from sklearn.base import clone
# Load dataset
data = np.load("data/processed_heartbeat_dataset.npz")

segments = data["segments"]
labels = data["grouped_labels"]
records = data["records"]
samples = data["samples"]
print("Dataset loaded!")
print("Total beats:", len(labels))
print("Total records:", len(np.unique(records)))
# -----------------------------------
# Build engineered biomedical features
# -----------------------------------

sampling_frequency = 360
baseline = np.median(segments, axis=1)

baseline_corrected = (
    segments - baseline[:, np.newaxis]
)


# Morphology features
# Estimate each beat's local baseline
baseline = np.median(
    segments,
    axis=1
)

# Center each heartbeat around its baseline
baseline_corrected = (
    segments - baseline[:, np.newaxis]
)


# Morphology features from baseline-corrected ECG
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

waveform_area = np.sum(
    np.abs(baseline_corrected),
    axis=1
) * dt

# Timing features
previous_rr = np.full(len(samples), np.nan)
next_rr = np.full(len(samples), np.nan)

for record_name in np.unique(records):

    record_indices = np.where(
        records == record_name
    )[0]

    record_samples = samples[record_indices]

    rr_intervals = (
        np.diff(record_samples)
        / sampling_frequency
    )

    previous_rr[record_indices[1:]] = (
        rr_intervals
    )

    next_rr[record_indices[:-1]] = (
        rr_intervals
    )


local_rr_average = (
    previous_rr + next_rr
) / 2

rr_ratio = (
    previous_rr / local_rr_average
)


# Replace missing timing values
previous_rr = np.nan_to_num(
    previous_rr,
    nan=0.0
)

next_rr = np.nan_to_num(
    next_rr,
    nan=0.0
)

rr_ratio = np.nan_to_num(
    rr_ratio,
    nan=1.0
)


# Combine features into one matrix
features = np.column_stack([
    max_voltage,
    min_voltage,
    voltage_range,
    rms_amplitude,
    waveform_area,
    previous_rr,
    next_rr,
    rr_ratio
])

print("\nFEATURE MATRIX")
print("Shape:", features.shape)



# -----------------------------------
# Find a better record-wise split
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

    test_labels = labels[test_idx]

    classes, counts = np.unique(
        test_labels,
        return_counts=True
    )

    class_counts = dict(zip(classes, counts))

    # Require useful representation of the rare classes
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


if best_split is None:
    raise ValueError(
        "No suitable record-wise split found."
    )


seed, train_indices, test_indices = best_split

print("Selected random seed:", seed)

X_train = features[train_indices]
X_test = features[test_indices]

y_train = labels[train_indices]
y_test = labels[test_indices]

train_records = records[train_indices]
test_records = records[test_indices]
# -----------------------------------
# Exclude Q beats from primary model
# -----------------------------------

train_keep = y_train != "Q"
test_keep = y_test != "Q"

X_train = X_train[train_keep]
y_train = y_train[train_keep]
train_records = train_records[train_keep]

X_test = X_test[test_keep]
y_test = y_test[test_keep]
test_records = test_records[test_keep]

print("\nPRIMARY 4-CLASS DATASET")
print("Training beats:", len(y_train))
print("Testing beats:", len(y_test))


print("\nRECORD-WISE SPLIT")

print("Training beats:", len(X_train))
print("Testing beats:", len(X_test))

print(
    "Training records:",
    len(np.unique(train_records))
)

print(
    "Testing records:",
    len(np.unique(test_records))
)


overlap = np.intersect1d(
    np.unique(train_records),
    np.unique(test_records)
)

print(
    "Records appearing in BOTH sets:",
    overlap
)
def print_class_distribution(name, labels):

    classes, counts = np.unique(
        labels,
        return_counts=True
    )

    print(f"\n{name} CLASS DISTRIBUTION")

    for heartbeat_class, count in zip(classes, counts):

        percentage = count / len(labels) * 100

        print(
            heartbeat_class,
            ":",
            count,
            "beats |",
            round(percentage, 2),
            "%"
        )


print_class_distribution(
    "TRAINING",
    y_train
)

print_class_distribution(
    "TESTING",
    y_test
)
# -----------------------------------
# Logistic Regression baseline model
# -----------------------------------

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])

print("\nTraining Logistic Regression model...")

model.fit(
    X_train,
    y_train
)

print("Training complete!")


# Make predictions
y_pred = model.predict(
    X_test
)


# Overall accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    "\nTEST ACCURACY:",
    round(accuracy * 100, 2),
    "%"
)


# Detailed metrics
print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        y_pred,
        digits=3
    )
)


# Confusion matrix
print("\nCONFUSION MATRIX")

print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=["N", "S", "V", "F"]
    )
)
# -----------------------------------
# Random Forest model
# -----------------------------------

rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest model...")

rf_model.fit(
    X_train,
    y_train
)

print("Random Forest training complete!")


rf_pred = rf_model.predict(
    X_test
)


rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

print(
    "\nRANDOM FOREST TEST ACCURACY:",
    round(rf_accuracy * 100, 2),
    "%"
)


print("\nRANDOM FOREST CLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        rf_pred,
        digits=3
    )
)


print("\nRANDOM FOREST CONFUSION MATRIX")

print(
    confusion_matrix(
        y_test,
        rf_pred,
        labels=["N", "S", "V", "F"]
    )
)
# -----------------------------------
# Random Forest feature importance
# -----------------------------------

feature_names = [
    "Maximum voltage",
    "Minimum voltage",
    "Voltage range",
    "RMS amplitude",
    "Waveform area",
    "Previous RR",
    "Next RR",
    "RR ratio"
]

importances = rf_model.feature_importances_

importance_order = np.argsort(
    importances
)[::-1]

print("\nRANDOM FOREST FEATURE IMPORTANCE")

for index in importance_order:

    print(
        feature_names[index],
        ":",
        round(importances[index], 4)
    )
    # -----------------------------------
# Error analysis by ECG record
# -----------------------------------

print("\nPER-RECORD RANDOM FOREST PERFORMANCE")

for record_name in np.unique(test_records):

    record_mask = test_records == record_name

    record_true = y_test[record_mask]
    record_pred = rf_pred[record_mask]

    record_accuracy = accuracy_score(
        record_true,
        record_pred
    )

    print(
        "\nRecord:",
        record_name,
        "| Beats:",
        len(record_true),
        "| Accuracy:",
        round(record_accuracy * 100, 2),
        "%"
    )

    true_classes, true_counts = np.unique(
        record_true,
        return_counts=True
    )

    print(
        "True classes:",
        dict(zip(true_classes, true_counts))
    )
    # -----------------------------------
# Save Random Forest confusion matrix
# -----------------------------------

disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    rf_pred,
    labels=["N", "S", "V", "F"],
    cmap="Blues",
    values_format="d"
)

plt.title(
    "Random Forest Confusion Matrix\n"
    "Record-Wise Test Set"
)

plt.tight_layout()

plt.savefig(
    "results/random_forest_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# -----------------------------------
# Final model comparison
# -----------------------------------

lr_balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

rf_balanced_accuracy = balanced_accuracy_score(
    y_test,
    rf_pred
)

lr_macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

rf_macro_f1 = f1_score(
    y_test,
    rf_pred,
    average="macro"
)


print("\nFINAL MODEL COMPARISON")

print("\nLogistic Regression")
print(
    "Accuracy:",
    round(
        accuracy_score(y_test, y_pred) * 100,
        2
    ),
    "%"
)
print(
    "Balanced accuracy:",
    round(lr_balanced_accuracy * 100, 2),
    "%"
)
print(
    "Macro F1:",
    round(lr_macro_f1, 3)
)


print("\nRandom Forest")
print(
    "Accuracy:",
    round(
        accuracy_score(y_test, rf_pred) * 100,
        2
    ),
    "%"
)
print(
    "Balanced accuracy:",
    round(rf_balanced_accuracy * 100, 2),
    "%"
)
print(
    "Macro F1:",
    round(rf_macro_f1, 3)
)
# -----------------------------------
# Grouped cross-validation
# TRAINING RECORDS ONLY
# -----------------------------------

print("\nGROUPED CROSS-VALIDATION")
print("Using training records only.")

group_kfold = GroupKFold(
    n_splits=5
)

cv_balanced_accuracies = []
cv_macro_f1_scores = []

fold_number = 1

for cv_train_idx, cv_val_idx in group_kfold.split(
    X_train,
    y_train,
    groups=train_records
):

    print(
        "\n------------------------------"
    )

    print(
        "FOLD",
        fold_number
    )

    print(
        "------------------------------"
    )

    X_cv_train = X_train[cv_train_idx]
    X_cv_val = X_train[cv_val_idx]

    y_cv_train = y_train[cv_train_idx]
    y_cv_val = y_train[cv_val_idx]

    # Fresh Random Forest for each fold
    cv_model = clone(rf_model)

    cv_model.fit(
        X_cv_train,
        y_cv_train
    )

    cv_pred = cv_model.predict(
        X_cv_val
    )

    # -----------------------------------
    # Per-class validation performance
    # -----------------------------------

    print("\nValidation class details:")

    for heartbeat_class in ["N", "S", "V", "F"]:

        class_mask = (
            y_cv_val == heartbeat_class
        )

        class_count = np.sum(
            class_mask
        )

        if class_count > 0:

            class_correct = np.sum(
                cv_pred[class_mask]
                == heartbeat_class
            )

            class_recall = (
                class_correct
                / class_count
            )

            print(
                heartbeat_class,
                "| Beats:",
                class_count,
                "| Correct:",
                class_correct,
                "| Recall:",
                round(
                    class_recall * 100,
                    2
                ),
                "%"
            )

        else:

            print(
                heartbeat_class,
                "| Beats: 0",
                "| Recall: not available"
            )

    # -----------------------------------
    # Overall fold metrics
    # -----------------------------------

    fold_balanced_accuracy = (
        balanced_accuracy_score(
            y_cv_val,
            cv_pred
        )
    )

    fold_macro_f1 = f1_score(
        y_cv_val,
        cv_pred,
        average="macro"
    )

    cv_balanced_accuracies.append(
        fold_balanced_accuracy
    )

    cv_macro_f1_scores.append(
        fold_macro_f1
    )

    print(
        "\nBalanced accuracy:",
        round(
            fold_balanced_accuracy * 100,
            2
        ),
        "%"
    )

    print(
        "Macro F1:",
        round(
            fold_macro_f1,
            3
        )
    )

    fold_number += 1


print("\n5-FOLD GROUPED CV SUMMARY")

print(
    "Mean balanced accuracy:",
    round(
        np.mean(
            cv_balanced_accuracies
        ) * 100,
        2
    ),
    "%"
)

print(
    "Balanced accuracy standard deviation:",
    round(
        np.std(
            cv_balanced_accuracies
        ) * 100,
        2
    ),
    "%"
)

print(
    "Mean macro F1:",
    round(
        np.mean(
            cv_macro_f1_scores
        ),
        3
    )
)

print(
    "Macro F1 standard deviation:",
    round(
        np.std(
            cv_macro_f1_scores
        ),
        3
    )
)
# -----------------------------------
# Logistic Regression grouped CV
# -----------------------------------

print("\nLOGISTIC REGRESSION GROUPED CROSS-VALIDATION")

lr_cv_balanced_accuracies = []
lr_cv_macro_f1_scores = []

fold_number = 1

for cv_train_idx, cv_val_idx in group_kfold.split(
    X_train,
    y_train,
    groups=train_records
):
    

    X_cv_train = X_train[cv_train_idx]
    X_cv_val = X_train[cv_val_idx]

    y_cv_train = y_train[cv_train_idx]
    y_cv_val = y_train[cv_val_idx]

    lr_cv_model = clone(model)

    lr_cv_model.fit(
        X_cv_train,
        y_cv_train
    )

    lr_cv_pred = lr_cv_model.predict(
        X_cv_val
    )

    fold_balanced_accuracy = balanced_accuracy_score(
        y_cv_val,
        lr_cv_pred
    )

    fold_macro_f1 = f1_score(
        y_cv_val,
        lr_cv_pred,
        average="macro"
    )

    lr_cv_balanced_accuracies.append(
        fold_balanced_accuracy
    )

    lr_cv_macro_f1_scores.append(
        fold_macro_f1
    )

    print(
        "\nFold",
        fold_number
    )

    print(
        "Balanced accuracy:",
        round(
            fold_balanced_accuracy * 100,
            2
        ),
        "%"
    )

    print(
        "Macro F1:",
        round(
            fold_macro_f1,
            3
        )
    )

    fold_number += 1


print("\nLOGISTIC REGRESSION 5-FOLD CV SUMMARY")

print(
    "Mean balanced accuracy:",
    round(
        np.mean(
            lr_cv_balanced_accuracies
        ) * 100,
        2
    ),
    "%"
)

print(
    "Balanced accuracy standard deviation:",
    round(
        np.std(
            lr_cv_balanced_accuracies
        ) * 100,
        2
    ),
    "%"
)

print(
    "Mean macro F1:",
    round(
        np.mean(
            lr_cv_macro_f1_scores
        ),
        3
    )
)

print(
    "Macro F1 standard deviation:",
    round(
        np.std(
            lr_cv_macro_f1_scores
        ),
        3
    )
)
# -----------------------------------
# Record diversity by heartbeat class
# -----------------------------------

print("\nRECORD DIVERSITY BY HEARTBEAT CLASS")

for heartbeat_class in ["N", "S", "V", "F"]:

    class_mask = (
        labels == heartbeat_class
    )

    class_records = np.unique(
        records[class_mask]
    )

    class_beat_count = np.sum(
        class_mask
    )

    print(
        "\nClass:",
        heartbeat_class
    )

    print(
        "Total beats:",
        class_beat_count
    )

    print(
        "Number of records containing class:",
        len(class_records)
    )

    print(
        "Records:",
        class_records
    )
    # -----------------------------------
# Save final evaluation figures
# -----------------------------------

# Model comparison figure
model_names = [
    "Logistic Regression",
    "Random Forest"
]

mean_balanced_accuracy = [
    np.mean(lr_cv_balanced_accuracies) * 100,
    np.mean(cv_balanced_accuracies) * 100
]

balanced_accuracy_std = [
    np.std(lr_cv_balanced_accuracies) * 100,
    np.std(cv_balanced_accuracies) * 100
]

plt.figure(figsize=(8, 5))

plt.bar(
    model_names,
    mean_balanced_accuracy,
    yerr=balanced_accuracy_std,
    capsize=8
)

plt.ylabel(
    "Balanced Accuracy (%)"
)

plt.title(
    "5-Fold Record-Grouped Cross-Validation"
)

plt.ylim(0, 65)

plt.tight_layout()

plt.savefig(
    "results/grouped_cv_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# -----------------------------------
# Random Forest feature importance
# -----------------------------------

feature_names = [
    "Maximum voltage",
    "Minimum voltage",
    "Voltage range",
    "RMS amplitude",
    "Waveform area",
    "Previous RR",
    "Next RR",
    "RR ratio"
]

importances = (
    rf_model.feature_importances_
)

sorted_indices = np.argsort(
    importances
)

plt.figure(figsize=(9, 6))

plt.barh(
    np.array(feature_names)[sorted_indices],
    importances[sorted_indices]
)

plt.xlabel(
    "Random Forest Feature Importance"
)

plt.title(
    "Physiology-Based ECG Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "results/random_forest_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nSaved final evaluation figures!"
)
# -----------------------------------
# Save final Random Forest model
# -----------------------------------

joblib.dump(
    rf_model,
    "results/random_forest_model.joblib"
)

print(
    "\nRandom Forest model saved successfully."
)