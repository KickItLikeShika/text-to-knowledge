import spacy

nlp = spacy.load("en_core_web_md")
nlp.add_pipe("entityLinker", last=False)


def named_entity_linking(text: str) -> dict:
    """
    Named Entity Linking on the given text using spacy entity linker
    https://github.com/egerber/spaCy-entity-linker

    Args:
        text(str): corefernce resolved text

    Returns:
        dict: {entity text: {entity type: entity type,
        named entity linking description: description, start char: start char,
        end char: end char}}
    """
    doc = nlp(text)
    linked_entities = {}
    for linked_entity in doc._.linkedEntities:
        linked_entities[linked_entity.label] = {
            "entity_type": linked_entity.get_super_entities()[0].label,
            "nel_description": linked_entity.description,
            "start": doc[linked_entity.get_span().start].idx,  # char start
            "end": doc[linked_entity.get_span().end].idx,  # char end
        }

    return linked_entities
