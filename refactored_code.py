import pandas as pd


AGE_THRESHOLD = 50
TARGET_CANCER_TYPE = "Lung"
TREATMENT_GROUP = "Treatment_A"
CONTROL_GROUP = "Control"


def load_clinical_data(file_path: str) -> pd.DataFrame:
    """
    Load clinical gene expression / patient metadata.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded clinical dataset.
    """
    return pd.read_csv(file_path)


def get_responsive_patients(data: pd.DataFrame) -> list[str]:
    """
    Identify patients older than AGE_THRESHOLD with lung cancer
    whose tumor size decreased after treatment.

    Parameters
    ----------
    data : pd.DataFrame
        Clinical dataset.

    Returns
    -------
    list[str]
        List of patient_id values for responsive patients.
    """
    filtered = data[
        (data["age"] >= AGE_THRESHOLD)
        & (data["cancer_type"] == TARGET_CANCER_TYPE)
        & (data["final_tumor_size"] < data["baseline_tumor_size"])
    ]

    return filtered["patient_id"].tolist()


def compute_survival_statistics(data: pd.DataFrame) -> dict:
    """
    Compute average survival time for each treatment group.

    Parameters
    ----------
    data : pd.DataFrame
        Clinical dataset.

    Returns
    -------
    dict
        Dictionary containing:
        {
            "avg_survival_treatment_group": float,
            "avg_survival_control": float
        }
    """
    grouped = data.groupby("treatment")["survival_months"].mean()

    return {
        "avg_survival_treatment_group": grouped.get(TREATMENT_GROUP, 0),
        "avg_survival_control": grouped.get(CONTROL_GROUP, 0),
    }


def analyze_clinical_trial(file_path: str) -> list[str]:
    """
    Run the full clinical trial analysis:

    Parameters
    ----------
    file_path : str
        Path to the clinical trial CSV file.

    Returns
    -------
    list[str]
        Patient IDs of responsive patients.
    """
    data = load_clinical_data(file_path)
    responsive_patients = get_responsive_patients(data)
    survival_stats = compute_survival_statistics(data)

    print("Avg survival (Treatment A):", survival_stats["avg_survival_treatment_group"])
    print("Avg survival (Control):", survival_stats["avg_survival_control"])
    print("Responsive patients:", len(responsive_patients))

    return responsive_patients


if __name__ == "__main__":
    result = analyze_clinical_trial("clinical_trial_patients.csv")
