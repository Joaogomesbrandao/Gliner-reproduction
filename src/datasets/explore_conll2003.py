from datasets import load_dataset

def main():
    dataset = load_dataset(
        "conll2003",
        cache_dir="../../data/raw",
        trust_remote_code=True
    )

    print("=" * 60)
    print("Quantidade de exemplos")
    print("=" * 60)

    print(f"Train      : {len(dataset['train'])}")
    print(f"Validation : {len(dataset['validation'])}")
    print(f"Test       : {len(dataset['test'])}")

    print("\n" + "=" * 60)
    print("NER Labels")
    print("=" * 60)

    ner_labels = dataset["train"].features["ner_tags"].feature.names

    for idx, label in enumerate(ner_labels):
        print(f"{idx}: {label}")

    print("\n" + "=" * 60)
    print("Primeira sentença")
    print("=" * 60)

    sample = dataset["train"][0]

    for token, tag in zip(sample["tokens"], sample["ner_tags"]):
        print(f"{token:<15} {ner_labels[tag]}")


if __name__ == "__main__":
    main()