import os
import shutil
import pandas as pd

BASE_PATH = "output/skill_profiles_explainable.csv"
PARSED_PATH = "output/skill_profile_parsed_single.csv"
BACKUP_PATH = "output/skill_profiles_explainable_backup.csv"


def main():
    if not os.path.exists(BASE_PATH):
        raise FileNotFoundError(f"Base profiles file not found: {BASE_PATH}")

    if not os.path.exists(PARSED_PATH):
        raise FileNotFoundError(f"Parsed profile file not found: {PARSED_PATH}")

    base = pd.read_csv(BASE_PATH)
    parsed = pd.read_csv(PARSED_PATH)

    if "StudentID" not in parsed.columns:
        raise ValueError("Parsed profile is missing 'StudentID' column")

    unique_ids = parsed["StudentID"].unique()
    if len(unique_ids) != 1:
        raise ValueError(f"Expected exactly 1 StudentID in parsed file, found: {unique_ids}")

    student_id = unique_ids[0]
    print(f"Merging parsed student {student_id} into base profiles")

    # Make a safety backup of the original file
    if not os.path.exists(BACKUP_PATH):
        shutil.copy(BASE_PATH, BACKUP_PATH)
        print(f"Backup written to {BACKUP_PATH}")
    else:
        print(f"Backup already exists at {BACKUP_PATH}")

    # If columns differ, align them
    for col in base.columns:
        if col not in parsed.columns:
            parsed[col] = None
    parsed = parsed[base.columns]  # same column order

    # Drop any old rows for this student (if present)
    base_no_student = base[base["StudentID"] != student_id]

    combined = pd.concat([base_no_student, parsed], ignore_index=True)

    print(f"Original rows: {len(base)}")
    print(f"New rows for {student_id}: {len(parsed)}")
    print(f"Combined rows: {len(combined)}")
    print(f"Unique students after merge: {combined['StudentID'].nunique()}")

    combined.to_csv(BASE_PATH, index=False)
    print(f"Updated base profiles written to {BASE_PATH}")


if __name__ == "__main__":
    main()
