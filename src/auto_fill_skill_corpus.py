import os
import pandas as pd

INPUT_PATH = "content/skill_corpus.csv"
OUTPUT_PATH = "content/skill_corpus_filled.csv"


def generate_content(skill: str) -> str:
    """
    Generate a short explanation paragraph for a given skill.
    For some key data-science / IT skills we give a custom description.
    For everything else we use a general template that is still
    good enough for RAG + quiz generation.
    """
    s = skill.strip()

    # ---- Hand-crafted descriptions for important skills ----
    custom = {
        "Programming Fundamentals & C Language":
            "C programming fundamentals including variables, control structures, "
            "functions, pointers, arrays and file handling. Students learn how to "
            "translate problem statements into working code, debug programs and "
            "follow basic coding standards.",

        "Object-Oriented Programming Fundamentals":
            "Core object-oriented concepts such as classes, objects, methods, "
            "encapsulation, inheritance and polymorphism. Focus on designing "
            "reusable components and modelling real-world problems in code.",

        "Java & OOP Implementation":
            "Applying object-oriented principles using Java, including class "
            "design, interfaces, collections, exceptions, threads and basic "
            "design patterns to build complete applications.",

        "Data Structures & Algorithms":
            "Design and analysis of fundamental data structures such as arrays, "
            "linked lists, stacks, queues and trees, together with searching, "
            "sorting and complexity analysis using asymptotic notation.",

        "Database Design & SQL":
            "Relational database design using ER modelling, normalization and "
            "schema refinement, followed by practical skills in SQL for creating "
            "tables, writing queries, joins, views and enforcing constraints.",

        "Web Development (HTML/CSS/JS/PHP)":
            "Full-stack web fundamentals including semantic HTML, styling with "
            "CSS, client-side scripting in JavaScript and server-side logic with "
            "PHP to build interactive, data-driven web applications.",

        "Computer Networks":
            "Concepts of computer networking including TCP/IP, addressing, "
            "subnetting, routing, switching, VLANs, ACLs and basic network "
            "security, with hands-on configuration of routers and switches.",

        "Operating Systems":
            "Core operating system concepts such as processes, threads, "
            "scheduling, synchronization, deadlocks, memory management, "
            "file systems and I/O, with examples from Unix-like systems.",

        "Probability & Statistics":
            "Foundation in probability, random variables, common distributions, "
            "sampling, estimation, hypothesis testing, regression and time "
            "series, with emphasis on interpreting outputs for real datasets.",

        "Data Warehousing & Business Intelligence":
            "Principles of data warehousing such as OLTP vs OLAP, star and "
            "snowflake schemas, ETL/ELT pipelines, OLAP cubes and BI reporting "
            "to support analytical decision-making.",

        "Machine Learning Foundations":
            "Supervised and unsupervised learning concepts, model training and "
            "evaluation, bias–variance trade-off, overfitting, regularization and "
            "practical workflow for building ML models.",

        "Neural Networks & Deep Learning":
            "Feed-forward neural networks with backpropagation, activation "
            "functions, convolutional and recurrent architectures, and practical "
            "issues such as optimization and regularization.",

        "Fundamentals of Data Mining":
            "Classical data-mining tasks such as classification, clustering, "
            "association rule mining and anomaly detection, with focus on how "
            "to select, train and evaluate models for business problems.",

        "Massive Data Processing & Cloud Computing":
            "Concepts and tools for large-scale data processing, including "
            "distributed storage, parallel processing frameworks and the use of "
            "cloud platforms for scalable analytics solutions.",

        "Information Retrieval & Web Analytics":
            "Techniques for indexing and retrieving documents, ranking search "
            "results, evaluating IR systems and analysing web traffic using "
            "metrics, logs and web analytics tools.",

        "Visual Analytics & User Experience Design":
            "Design of effective visualizations and dashboards, principles of "
            "perception and UI/UX, and use of visual analytics to explore and "
            "communicate patterns in data.",

        "Introduction to Information Security Analytics":
            "Fundamental security concepts such as threats, vulnerabilities and "
            "controls, together with basic analytical techniques for log "
            "analysis, anomaly detection and security monitoring.",

        "Database Administration & Storage Systems":
            "Administration of enterprise database servers including installation, "
            "configuration, backup and recovery, access control, performance "
            "tuning and understanding of storage architectures such as RAID, "
            "SAN and cloud storage.",

        "Agile Development & SCRUM":
            "Agile principles and Scrum practices including roles, ceremonies, "
            "backlog management, iterative delivery and how to collaborate in an "
            "Agile software team.",

        "Academic Writing & Grammar":
            "Skills for structuring academic writing, using formal style, correct "
            "grammar and cohesion, and integrating citations and references.",
        
        "Academic Listening & Note-taking":
            "Listening strategies for lectures and academic talks, with emphasis "
            "on identifying key ideas, recording structured notes and extracting "
            "relevant information for later study.",
    }

    if s in custom:
        return custom[s]

    # ---- Generic templates based on keywords ----
    lower = s.lower()

    if "probability" in lower or "statistics" in lower or "hypothesis" in lower:
        return (
            f"{s} involves using statistical thinking to analyse data, quantify "
            "uncertainty and make evidence-based decisions, including choosing "
            "appropriate tests, interpreting p-values and confidence intervals."
        )

    if "regression" in lower:
        return (
            f"{s} covers building and interpreting regression models, checking "
            "assumptions, diagnosing issues and using the models for prediction "
            "and explanation."
        )

    if "neural" in lower or "deep" in lower or "machine learning" in lower:
        return (
            f"{s} focuses on learning patterns from data using parameterised "
            "models, training procedures, evaluation metrics and regularization "
            "to generalize to unseen cases."
        )

    if "sql" in lower or "database" in lower or "relational" in lower:
        return (
            f"{s} covers designing relational schemas and using SQL to define, "
            "manipulate and query data while preserving integrity and "
            "performance."
        )

    if "data structure" in lower or "algorithm" in lower:
        return (
            f"{s} focuses on efficient organisation of data and step-by-step "
            "procedures for solving computational problems, with attention to "
            "time and space complexity."
        )

    if "network" in lower:
        return (
            f"{s} deals with how devices communicate over networks, including "
            "addressing, routing, switching, protocols and basic security controls."
        )

    if "operating system" in lower or "process" in lower or "thread" in lower:
        return (
            f"{s} relates to how an operating system manages processes, threads, "
            "memory, files and I/O devices to provide a safe and efficient "
            "execution environment."
        )

    if "web" in lower or "http" in lower or "html" in lower or "javascript" in lower:
        return (
            f"{s} is about building and delivering content and applications over "
            "the web, including front-end interfaces and server-side processing."
        )

    if "big data" in lower or "data warehouse" in lower or "olap" in lower:
        return (
            f"{s} focuses on storing large volumes of data for analytics, "
            "designing dimensional models and supporting fast, ad-hoc queries "
            "for business decision-making."
        )

    if "visual" in lower or "analytics" in lower or "dashboard" in lower:
        return (
            f"{s} covers designing clear charts and dashboards that help users "
            "understand patterns, trends and anomalies in data."
        )

    if "security" in lower or "encryption" in lower or "acl" in lower:
        return (
            f"{s} is concerned with protecting systems and data through access "
            "control, secure configuration and detection of suspicious activity."
        )

    if "professional" in lower or "team" in lower or "soft" in lower:
        return (
            f"{s} refers to communication, teamwork, ethics and other "
            "professional behaviours required to work effectively in industry."
        )

    # ---- Fallback generic description ----
    return (
        f"{s} is a core skill area in the IT / data science curriculum. It covers "
        "the key concepts, typical tools and practical problem-solving abilities "
        "needed to apply this knowledge in real projects."
    )


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Cannot find {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    if "Content" not in df.columns:
        raise ValueError("Expected a 'Content' column in skill_corpus.csv")

    filled_contents = []
    for _, row in df.iterrows():
        skill = str(row["Skill"])
        current = str(row.get("Content", "")).strip()
        if current and current.lower() != "nan":
            # keep whatever you already typed
            filled_contents.append(current)
        else:
            filled_contents.append(generate_content(skill))

    df["Content"] = filled_contents

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Written filled corpus to {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
