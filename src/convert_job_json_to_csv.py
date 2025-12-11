import json
import os
from urllib.parse import urlparse

import pandas as pd

# Paths inside your project
INPUT_JSON = "input/Job_data.json"           # put Job_data.json here
OUTPUT_CSV = "input/job_postings_sample.csv" # this will replace the old sample CSV


def detect_source(job_url: str) -> str:
    """
    Infer the source (LinkedIn, TopJobs, etc.) from the job URL.
    """
    if not isinstance(job_url, str) or not job_url:
        return ""

    netloc = urlparse(job_url).netloc.lower()

    if "linkedin" in netloc:
        return "LinkedIn"
    if "topjobs" in netloc:
        return "TopJobs"

    return netloc or "Unknown"


def main():
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(
            f"Cannot find {INPUT_JSON}. "
            f"Make sure Job_data.json is saved as {INPUT_JSON}"
        )

    # 1) Load JSON
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected Job_data.json to contain a list of job objects")

    df = pd.json_normalize(data)

    # 2) Map JSON fields -> CSV columns used in your old pipeline
    # JSON keys we saw: job_id, title, company, location, posted_date, job_url, description, ...
    out = pd.DataFrame(
        {
            "JobID": df.get("job_id"),
            "Title": df.get("title"),
            "Company": df.get("company"),
            "Location": df.get("location"),
            "Description": df.get("description"),
            "Source": df.get("job_url").apply(detect_source),
            "PostedDate": df.get("posted_date"),
        }
    )

    # 3) Save CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"Wrote {len(out)} rows to {OUTPUT_CSV}")
    print("Columns:", list(out.columns))


if __name__ == "__main__":
    main()
