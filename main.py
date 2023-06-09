from pipeline.en.coreference_resolution import coref_resolution
from pipeline.en.named_entity_recognition import named_entity_recognition
from pipeline.en.relationship_extraction import relationship_extraction
from pipeline.en.knowledge_graph import build_knowledge_graph


# text = """Elon Musk is a business magnate, industrial designer, and engineer. He is the founder, 
#         CEO, CTO, and chief designer of SpaceX. He is also early investor, CEO, and product architect of Tesla, Inc. 
#         He is also the founder of The Boring Company and the co-founder of Neuralink. A centibillionaire, 
#         Musk became the richest person in the world in January 2021, with an estimated net worth of $185 billion at the time, 
#         surpassing Jeff Bezos. Musk was born to a Canadian mother and South African father and raised in Pretoria, South Africa. 
#         He briefly attended the University of Pretoria before moving to Canada aged 17 to attend Queen's University. 
#         He transferred to the University of Pennsylvania two years later, where he received dual bachelor's degrees in economics and physics. 
#         He moved to California in 1995 to attend Stanford University, but decided instead to pursue a business career. 
#         He went on co-founding a web software company Zip2 with his brother Kimbal Musk."""

text = """Elon Musk is a business magnate, industrial designer, and engineer. He is the founder of SpaceX. 
        He is also early investor, CEO, and product architect of Tesla, Inc.
        He is also the founder of The Boring Company and the co-founder of Neuralink."""

coref_resolved_text = coref_resolution(text)
print("Corefernce Resolution done")

linked_entities = named_entity_recognition(coref_resolved_text)
print("Named Entity Linking done")

extracted_relationships = relationship_extraction(coref_resolved_text, linked_entities)
print("Relationship extraction done")

build_knowledge_graph(extracted_relationships)
