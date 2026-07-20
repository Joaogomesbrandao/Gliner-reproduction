from gliner import GLiNER

class GLiNERModel:
    def __init__(self, checkpoint="urchade/gliner_large-v2.1"):
        print(f"\nLoading {checkpoint}...\n")
        self.model = GLiNER.from_pretrained(checkpoint)

    def predict(self, text, labels, threshold=0.5):
        return self.model.predict_entities(
            text=text,
            labels=labels,
            threshold=threshold
        )