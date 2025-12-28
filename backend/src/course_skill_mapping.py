import pandas as pd
from typing import Dict, List


def load_course_skill_mapping(path: str) -> Dict[str, dict]:
    """
    Load the course-to-skill mapping CSV and return a dictionary.

    Returns:
        {
          "IT1010": {
              "title": "Introduction to Programming",
              "skills": ["Procedural Programming Concepts", ...],
              "main_skill": "Programming Fundamentals & C Language",
              "skill_level": "Beginner",
          },
          ...
        }
    """
    # Try multiple encodings to avoid UnicodeDecodeError on non-UTF-8 CSVs
    df = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        # Let pandas raise a helpful error if all encodings fail
        df = pd.read_csv(path)

    mapping: Dict[str, dict] = {}

    for _, row in df.iterrows():
        code = str(row["CourseCode"]).strip()

        skills: List[str] = []
        for col in ["Skill1", "Skill2", "Skill3", "Skill4", "Skill5"]:
            if col in row and isinstance(row[col], str) and row[col].strip():
                skills.append(row[col].strip())

        mapping[code] = {
            "title": row["CourseTitle"].strip(),
            "skills": skills,
            "main_skill": str(row["MainSkill"]).strip(),
            "skill_level": str(row["SkillLevel"]).strip(),
        }

    return mapping


if __name__ == "__main__":
    # quick manual test
    mapping = load_course_skill_mapping("input/course_skill_mapping.csv")
    print(f"Loaded {len(mapping)} courses")
    # print one sample
    print("IT1010 →", mapping.get("IT1010"))
    