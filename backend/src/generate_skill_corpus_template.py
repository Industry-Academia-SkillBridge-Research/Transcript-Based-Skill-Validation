import os
import pandas as pd

MAPPING_PATH = "input/course_skill_mapping.csv"
OUTPUT_DIR = "content"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "skill_corpus.csv")


def read_mapping_robust(path: str) -> pd.DataFrame:
    """
    Read course_skill_mapping.csv with multiple encodings,
    to avoid UnicodeDecodeError.
    """
    df = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            # print(f"Loaded mapping with encoding: {enc}")
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        # last attempt, let pandas raise a clear error
        df = pd.read_csv(path)

    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Load mapping with robust encoding
    df = read_mapping_robust(MAPPING_PATH)

    # 2) Collect unique skills from Skill1..Skill5 columns
    skill_cols = ["Skill1", "Skill2", "Skill3", "Skill4", "Skill5"]

    skills = set()
    for col in skill_cols:
        if col in df.columns:
            for x in df[col].dropna().tolist():
                text = str(x).strip()
                if text:
                    skills.add(text)

    skills = sorted(skills)

    # 3) Build template DataFrame
    corpus_df = pd.DataFrame(
        {
            "Skill": skills,
            "SourceType": ["module_outline"] * len(skills),
            "SourceName": ["AUTO_FROM_MAPPING"] * len(skills),
            "Content": ["" for _ in skills],  # you will fill this later
        }
    )

    # 4) Backup if file already exists
    if os.path.exists(OUTPUT_PATH):
        backup_path = OUTPUT_PATH.replace(".csv", "_backup.csv")
        os.replace(OUTPUT_PATH, backup_path)
        print(f"Existing skill_corpus.csv backed up to {backup_path}")

    corpus_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Template corpus created with {len(skills)} skills at {OUTPUT_PATH}")
    print("Now open it in Excel and fill the 'Content' column for the most important skills.")


if __name__ == "__main__":
    main()
