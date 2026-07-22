from __future__ import annotations

from src.datasets.conll2003 import DEFAULT_GLINER_LABELS, build_sentence, get_ner_labels, load_conll2003
from src.evaluation.bio_converter import bio_tags_to_spans, predictions_to_bio, predictions_to_token_spans
from src.models.gliner_model import GLiNERModel


def main() -> None:
    print("=" * 80)
    print("EXPERIMENTO QUALITATIVO - GLiNER ZERO-SHOT")
    print("=" * 80)

    dataset = load_conll2003()
    sentence = build_sentence(dataset["test"][0], get_ner_labels(dataset))

    print("\nTexto reconstruído:\n")
    print(sentence.text)

    print("\nGold spans:")
    print(bio_tags_to_spans(sentence.gold_bio_tags))

    model = GLiNERModel("urchade/gliner_large-v2.1")
    predictions = model.predict(
        text=sentence.text,
        labels=DEFAULT_GLINER_LABELS,
        threshold=0.5,
    )

    print("\nLabels fornecidas ao GLiNER:")
    print(list(DEFAULT_GLINER_LABELS))

    print("\n" + "=" * 80)
    print("ENTIDADES ENCONTRADAS")
    print("=" * 80)
    if not predictions:
        print("Nenhuma entidade encontrada.")
    else:
        for entity in predictions:
            print(entity)

    predicted_spans = predictions_to_token_spans(predictions, sentence.offsets)
    predicted_bio = predictions_to_bio(
        tokens=sentence.tokens,
        predictions=predictions,
        token_offsets=sentence.offsets,
    )

    print("\nPredicted spans:")
    print(predicted_spans)

    print("\n" + "=" * 80)
    print("COMPARAÇÃO ENTRE GOLD E PREDIÇÃO")
    print("=" * 80)
    print(f"{'TOKEN':20} {'GROUND TRUTH':15} {'PREDIÇÃO'}")
    print("-" * 55)

    for token, gold_tag, pred_tag in zip(
        sentence.tokens,
        sentence.gold_bio_tags,
        predicted_bio,
    ):
        print(f"{token:20} {gold_tag:15} {pred_tag}")


if __name__ == "__main__":
    main()
