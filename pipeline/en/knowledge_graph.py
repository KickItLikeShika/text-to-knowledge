import matplotlib.pyplot as plt
import networkx as nx


def build_knowledge_graph(extracted_relationships: dict):
    """Build a knowledge graph from a dict of extracted relationships between entities

    Args:
        extracted_relationships (dict): extracted relationships between source and target entities
    """
    G = nx.DiGraph()
    vertices = list(map(lambda source_entity: [source_entity, extracted_relationships[source_entity]['target_entity']], extracted_relationships.keys()))
    G.add_edges_from(vertices)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 9))
    nx.draw(G, pos, edge_color='black', width=1, linewidths=1, node_size=1200, node_color='red', alpha=0.9, labels={node: node for node in G.nodes()})

    relationship_types = {}
    for edge in G.edges():
        relationship_type = extracted_relationships[edge[0]]['relationship']
        relationship_types[edge] = relationship_type

    nx.draw_networkx_edge_labels(G, pos, edge_labels=relationship_types, font_color='red')
    plt.axis('off')
    plt.show()
