# step 1 
    # identify the following things in your dataset:

    # Nodes (Items, People, Locations, Cars, ...)
    # Edges (Connections, Interactions, Similarity, ...)
    # Node Features (Attributes)
    # Labels (Node-level, edge-level, graph-level)
    # and optionally:
        # Edge weights (Strength of the connection, number of interactions, ...)
        # Edge features (Additional (multi-dim) properties describing the edge)

# step 2
    # Do you have different node and edge types? (This means the nodes/edges have different attributes such as Cars vs. People)
        # No, all my edges/nodes have the same type --> Proceed with homogenious
        # Yes, there are different relations and node types --> Proceed with heterogenious

