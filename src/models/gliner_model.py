from __future__ import annotations

from typing import Any, Sequence

from gliner import GLiNER


class GLiNERModel:
    """Thin wrapper around the public GLiNER inference API."""

    def __init__(
        self,
        checkpoint: str = "urchade/gliner_large-v2.1",
        *,
        flat_ner: bool = True,
        model_name: str | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.flat_ner = flat_ner
        self.model_name = model_name or checkpoint
        print(f"\nLoading {checkpoint}...\n")
        self.model = GLiNER.from_pretrained(checkpoint)

    def predict(
        self,
        text: str,
        labels: Sequence[str],
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        return self.model.predict_entities(
            text=text,
            labels=list(labels),
            flat_ner=self.flat_ner,
            threshold=threshold,
        )

    def predict_batch(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        threshold: float = 0.5,
    ) -> list[list[dict[str, Any]]]:
        return self.model.batch_predict_entities(
            texts=list(texts),
            labels=list(labels),
            flat_ner=self.flat_ner,
            threshold=threshold,
        )
