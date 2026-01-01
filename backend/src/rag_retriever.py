import os

def load_skill_context(skill_key: str, kb_dir: str = "knowledge_base", max_chars: int = 5500) -> str:
    """
    Reads knowledge_base/<skill_key>.txt (lowercased).
    Example: SQL -> knowledge_base/sql.txt
    If not found, combines all .txt as fallback.
    """
    os.makedirs(kb_dir, exist_ok=True)

    fname = f"{skill_key.lower()}.txt"
    direct = os.path.join(kb_dir, fname)

    if os.path.exists(direct):
        with open(direct, "r", encoding="utf-8", errors="replace") as f:
            return f.read()[:max_chars]

    # fallback: combine all .txt
    parts = []
    for fn in os.listdir(kb_dir):
        if fn.lower().endswith(".txt"):
            with open(os.path.join(kb_dir, fn), "r", encoding="utf-8", errors="replace") as f:
                parts.append(f.read())

    return ("\n\n".join(parts))[:max_chars]
