from src.datasets.conll2003 import load_conll2003, get_ner_labels
from src.models.gliner_model import GLiNERModel
from src.evaluation.bio_converter import predictions_to_bio

GLINER_LABELS = [
    "person",
    "organization",
    "location",
    "miscellaneous"
]

def main():
    print("=" * 80)
    print("EXPERIMENTO 1 - ZERO-SHOT NER COM GLiNER")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Carrega o dataset
    # ------------------------------------------------------------------
    dataset = load_conll2003()

    # Primeira sentença do conjunto de teste
    sample = dataset["test"][0]

    tokens = sample["tokens"]
    text = " ".join(tokens)

    print("\nTexto:\n")
    print(text)

    # ------------------------------------------------------------------
    # Ground Truth
    # ------------------------------------------------------------------
    ner_labels = get_ner_labels(dataset)

    ground_truth = [
        ner_labels[tag]
        for tag in sample["ner_tags"]
    ]

    # ------------------------------------------------------------------
    # Carrega o modelo
    # ------------------------------------------------------------------
    model = GLiNERModel()

    print("\nLabels fornecidas ao GLiNER:")
    print(GLINER_LABELS)

    # ------------------------------------------------------------------
    # Inferência
    # ------------------------------------------------------------------
    predictions = model.predict(
        text=text,
        labels=GLINER_LABELS
    )

    print("\n" + "=" * 80)
    print("ENTIDADES ENCONTRADAS")
    print("=" * 80)

    if len(predictions) == 0:
        print("Nenhuma entidade encontrada.")
    else:
        for entity in predictions:
            print(entity)

    # ------------------------------------------------------------------
    # Conversão para BIO
    # ------------------------------------------------------------------
    predicted_bio = predictions_to_bio(
        tokens=tokens,
        predictions=predictions
    )

    # ------------------------------------------------------------------
    # Comparação
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPARAÇÃO ENTRE GROUND TRUTH E PREDIÇÃO")
    print("=" * 80)

    print(f"{'TOKEN':20} {'GROUND TRUTH':15} {'PREDIÇÃO'}")
    print("-" * 55)

    for token, gt, pred in zip(tokens, ground_truth, predicted_bio):
        print(f"{token:20} {gt:15} {pred}")

    print("\nFim do experimento.")


if __name__ == "__main__":
    main()