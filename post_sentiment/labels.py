LABELS = ("negative", "neutral", "positive")

def parse_label(value):
    label = value.strip().lower()
    if label not in LABELS:
        raise ValueError(f"Expected one of {', '.join(LABELS)}")
    return label
