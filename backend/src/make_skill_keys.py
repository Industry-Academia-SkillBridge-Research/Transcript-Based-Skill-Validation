import os
import re
import pandas as pd


# Keyword buckets (edit anytime)
BUCKETS = [
    (r"\bpython\b", "Python"),
    (r"\bjava\b", "Java"),
    (r"\bc\+\+\b|\bcpp\b", "C++"),
    (r"\bc\b(?!\+\+)", "C"),  # careful: only if you really need C as a bucket
    (r"\bsql\b|database|dbms|jdbc|odbc|schema|normalization|query|transaction|indexing", "SQL"),
    (r"statistics|statistical|probability|hypothesis|regression|bayesian|distribution|chi[-\s]?square|anova|t[-\s]?test", "Statistics"),
    (r"machine\s*learning|\bml\b|neural|cnn|rnn|lstm|perceptron|reinforcement|q[-\s]?learning|classification|overfitting|optimizer|gradient", "MachineLearning"),
    (r"data\s*visuali[sz]ation|visuali[sz]ation|chart|plot|dashboard|matplotlib|seaborn|tableau|power\s*bi", "DataVisualization"),
    (r"web|html|css|javascript|php|servlet|jsp|rest|api|angular|typescript", "WebDevelopment"),
    (r"network|osi|tcp|udp|routing|subnet|vlan|switch|router|acl", "Networking"),
    (r"operating\s*system|os\s|process|thread|memory|virtual\s*memory|file\s*system", "OperatingSystems"),
    (r"software\s*engineering|uml|requirements|testing|scrum|agile|design\s*pattern|configuration|version|release", "SoftwareEngineering"),
    (r"security|vulnerability|intrusion|risk|threat|access\s*control|crypt", "Security"),
    (r"math|mathematics|calculus|differentiation|integration|algebra|logic|graph\s*theory|matrices|probability", "Mathematics"),
]


def normalize_skill(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("�", "")  # remove bad chars if present
    s = re.sub(r"\s+", " ", s)
    return s


def make_fallback_key(skill: str) -> str:
    """
    If no bucket matches, create a compact key from words.
    Example: 'Academic Writing & Grammar' -> 'AcademicWriting'
    """
    s = normalize_skill(skill)
    s = re.sub(r"[^\w\s]", " ", s)
    words = [w for w in s.split() if w]
    if not words:
        return "Other"
    # take first 2 words, join in CamelCase
    take = words[:2]
    return "".join(w[:1].upper() + w[1:].lower() for w in take)


def skill_to_key(skill: str) -> str:
    s = normalize_skill(skill).lower()
    for pattern, key in BUCKETS:
        if re.search(pattern, s):
            return key
    return make_fallback_key(skill)


def main():
    in_path = os.path.join("input", "course_skill_mapping.csv")
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Not found: {in_path}")

    df = pd.read_csv(in_path)

    skill_cols = [c for c in ["Skill1", "Skill2", "Skill3", "Skill4", "Skill5", "MainSkill"] if c in df.columns]
    if not skill_cols:
        raise ValueError("No Skill1..Skill5/MainSkill columns found in course_skill_mapping.csv")

    # Collect unique skill names
    skills = set()
    for c in skill_cols:
        skills.update(df[c].dropna().astype(str).map(normalize_skill).tolist())

    skills = sorted(s for s in skills if s)

    out = pd.DataFrame(
        {
            "Skill": skills,
            "SkillKey": [skill_to_key(s) for s in skills],
        }
    )

    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", "skill_key_map_all.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(out)} skills)")


if __name__ == "__main__":
    main()
