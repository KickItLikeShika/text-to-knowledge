import spacy

nlp = spacy.load("en_core_web_md")


def named_entity_recognition(text: str) -> dict:
    """
    Named Entity Recognition on the given text

    Args:
        text(str): corefernce resolved text

    Returns:
        dict: {entity text: {entity type: entity type,
        named entity linking description: description, start char: start char,
        end char: end char}}
    """
    doc = nlp(text)

    entities = {}
    for ent in doc.ents:
        if ent.text in list(entities.keys()):
            continue

        entities[ent.text] = {
            "entity_type": ent.label_,
            "nel_description": "description",
            "start": doc[ent.start].idx, # char start
            "end": doc[ent.end].idx, # char end
        }

    return entities
