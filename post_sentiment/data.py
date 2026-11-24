import csv
from .labels import parse_label
from .text import normalise

def load(path):
    records = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(reader.fieldnames) != {"text", "label"} or len(reader.fieldnames) != 2:
            raise ValueError("CSV requires exactly two columns: text and label")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Wrong number of fields at CSV line {reader.line_num}")
            text = normalise(row["text"])
            if not text:
                raise ValueError(f"Empty text at CSV line {reader.line_num}")
            label = parse_label(row["label"])
            if text in records and records[text] != label:
                raise ValueError(f"Conflicting labels for a duplicate at CSV line {reader.line_num}")
            records[text] = label
    return list(records), list(records.values())
