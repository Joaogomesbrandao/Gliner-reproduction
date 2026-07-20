from src.datasets.conll2003 import build_sentence, get_ner_labels, load_conll2003


def main() -> None:
    dataset = load_conll2003()
    ner_labels = get_ner_labels(dataset)

    print("=" * 60)
    print("Quantidade de exemplos")
    print("=" * 60)
    print(f"Train      : {len(dataset['train'])}")
    print(f"Validation : {len(dataset['validation'])}")
    print(f"Test       : {len(dataset['test'])}")

    print("\n" + "=" * 60)
    print("NER Labels")
    print("=" * 60)
    for idx, label in enumerate(ner_labels):
        print(f"{idx}: {label}")

    sentence = build_sentence(dataset["train"][0], ner_labels)

    print("\n" + "=" * 60)
    print("Primeira sentença")
    print("=" * 60)
    print(sentence.text)

    print("\n" + "=" * 60)
    print("Tokens / BIO")
    print("=" * 60)
    for token, tag in zip(sentence.tokens, sentence.gold_bio_tags):
        print(f"{token:<15} {tag}")


if __name__ == "__main__":
    main()
