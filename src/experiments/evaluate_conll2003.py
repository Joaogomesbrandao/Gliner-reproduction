import time
from tqdm import tqdm

from src.datasets.conll2003 import (
    load_conll2003,
    get_ner_labels
)

from src.models.gliner_model import GLiNERModel

from src.evaluation.bio_converter import predictions_to_bio
from src.evaluation.metrics import evaluate_predictions


GLINER_LABELS = [
    "PER",
    "ORG",
    "LOC",
    "MISC"
]


def run_zero_shot_experiment(
    checkpoint="urchade/gliner_large-v2.1",
    labels=GLINER_LABELS,
    split="test",
    max_examples=None
):

    print("=" * 80)
    print("GLiNER ZERO-SHOT NER")
    print("=" * 80)

    print(f"Checkpoint : {checkpoint}")
    print(f"Split      : {split}")
    print(f"Labels     : {labels}")

    dataset = load_conll2003()
    ner_labels = get_ner_labels(dataset)

    samples = dataset[split]

    if max_examples is not None:
        samples = samples.select(range(max_examples))

    print(f"\nNúmero de exemplos: {len(samples)}")

    model = GLiNERModel(checkpoint)

    y_true = []
    y_pred = []

    total_time = 0

    for sample in tqdm(samples):

        tokens = sample["tokens"]

        text = " ".join(tokens)

        start = time.perf_counter()

        predictions = model.predict(
            text=text,
            labels=labels,
            threshold=0.5
        )

        total_time += time.perf_counter() - start

        pred_bio = predictions_to_bio(
            tokens=tokens,
            predictions=predictions
        )

        gt_bio = [
            ner_labels[tag]
            for tag in sample["ner_tags"]
        ]

        y_true.append(gt_bio)
        y_pred.append(pred_bio)

    metrics = evaluate_predictions(
        y_true,
        y_pred
    )

    print("\n" + "=" * 80)
    print("TEMPO")
    print("=" * 80)

    print(f"Tempo total     : {total_time:.2f} s")
    print(f"Tempo médio     : {total_time / len(samples):.4f} s")

    return {
        "checkpoint": checkpoint,
        "labels": labels,
        "examples": len(samples),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "time": total_time,
    }


def main():

    results = run_zero_shot_experiment()

    print("\n" + "=" * 80)
    print("RESUMO")
    print("=" * 80)

    for key, value in results.items():
        print(f"{key:<12}: {value}")


if __name__ == "__main__":
    main()