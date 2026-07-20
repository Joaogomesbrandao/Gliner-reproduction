from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "raw"

def load_conll2003():
    return load_dataset(
        "conll2003",
        cache_dir=str(CACHE_DIR),
        trust_remote_code=True,
    )

def get_ner_labels(dataset):
    return dataset["train"].features["ner_tags"].feature.names

if __name__ == "__main__":
    dataset = load_conll2003()
    print(dataset)
    print()
    print(get_ner_labels(dataset))