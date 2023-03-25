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
        for ent in doc.ents:
            if ent.text == linked_entity.label:
                linked_entities[ent.text] = {
                    "entity_type": ent.label_, 
                    "nel_description": linked_entity.description, 
                    "start": doc[ent.start].idx, # char start
                    "end": doc[ent.end].idx # char end
                }

    return linked_entities
