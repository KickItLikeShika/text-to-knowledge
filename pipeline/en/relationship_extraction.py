import opennre
import torch

relation_model = opennre.get_model("wiki80_bert_softmax")
if torch.cuda.is_available():
    relation_model = relation_model.cuda()


def relationship_extraction(text: str, linked_entities: dict):
    """_summary_

    Args:
        text(str): text
        linked_entities (dict): dict containing entities with types, NEL description and char start and end

    Returns:
        (dict): extracted relationships between source and target entities

    """
    extracted_relationships = {}
    for source_entity, source_value in linked_entities.items():
        for target_entity, target_value in linked_entities.items():
            if source_entity == target_entity:
                continue

            relationship = relation_model.infer(
                {
                    "text": text,
                    "h": {"pos": [source_value["start"], source_value["end"]]},
                    "t": {"pos": [target_value["start"], target_value["end"]]},
                }
            )

            if relationship[1] >= 0.5:  # relationship threshold
                extracted_relationships[source_entity] = {
                    "target_entity": target_entity,
                    "source_entity_type": source_value["entity_type"],
                    "target_entity_type": target_value["entity_type"],
                    "relationship": relationship[0],
                    "source_nel_description": source_value["nel_description"],
                    "target_nel_description": target_value["nel_description"],
                }

    return extracted_relationships
