from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def evaluate_predictions(y_true, y_pred):
    """
    Calcula Precision, Recall e F1.

    Parameters
    ----------
    y_true : List[List[str]]
        Tags BIO verdadeiras.

    y_pred : List[List[str]]
        Tags BIO previstas.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n================ RESULTADOS ================\n")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    print("\n============== RELATÓRIO ==============\n")
    print(classification_report(y_true, y_pred))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }