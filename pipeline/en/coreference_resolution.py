from allennlp.predictors.predictor import Predictor

model_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
predictor = Predictor.from_path(model_url)


def coref_resolution(text: str) -> str:
    """Given a piece of text, return a coreference resolved text.

    Args:
        text (str): text

    Returns:
        str: coreference resolved text
    """
    if type(text) is not str:
        raise TypeError("text must be a string")

    return predictor.coref_resolved(text)
