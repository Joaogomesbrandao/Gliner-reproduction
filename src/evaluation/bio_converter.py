from typing import List, Dict

LABEL_MAP = {
    # Labels naturais
    "person": "PER",
    "organization": "ORG",
    "location": "LOC",
    "miscellaneous": "MISC",

    # Labels do CoNLL
    "per": "PER",
    "org": "ORG",
    "loc": "LOC",
    "misc": "MISC",
}


def tokens_with_offsets(tokens: List[str]):
    offsets = []

    current = 0

    for token in tokens:
        start = current
        end = start + len(token)

        offsets.append((start, end))

        current = end + 1

    return offsets


def predictions_to_bio(tokens: List[str], predictions: List[Dict]):

    bio_tags = ["O"] * len(tokens)

    offsets = tokens_with_offsets(tokens)

    for prediction in predictions:

        label = prediction["label"].lower()

        if label not in LABEL_MAP:
            continue

        entity = LABEL_MAP[label]

        entity_start = prediction["start"]
        entity_end = prediction["end"]

        first = True

        for idx, (token_start, token_end) in enumerate(offsets):

            overlap = (
                token_start < entity_end
                and token_end > entity_start
            )

            if not overlap:
                continue

            if first:
                bio_tags[idx] = f"B-{entity}"
                first = False
            else:
                bio_tags[idx] = f"I-{entity}"

    return bio_tags