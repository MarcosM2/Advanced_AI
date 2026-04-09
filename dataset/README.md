# PERCI Sample Dataset

PERCI contains a sample dataset which may be used as a template for adaptations of the application. This dataset is based on parts of the ground floor and second floor of Arás an Phiarsaigh at Trinity College Dublin, although except with consent, the names of specific individuals have been replaced with pseudonyms.

The dataset is composed of a series of nodes, which either represent potential destinations or intermediate points used to reach them, such as corners and junctions. By defining intermediate points, PERCI can respond in a manner similar to how a human may give directions. These nodes are connected by edges, which are assigned weights based on the distance between the nodes.

Node structure:

    class Node:
        def __init__(self, node_id: str, name: str, x: float, y: float):
            self.id = node_id
            self.name = name
            self.x = x
            self.y = y

Example Node:

    "entrance": Node("entrance", "Entrance", 0, 0)

- `id`: An identifier for the node for use with the pathfinding algorithm\
- `name`: A human-readable description of the node\
- `x`, `y`: `x` and `y` coordinates for the node in a grid

Example Edge:

    ("entrance", "lobby", 4, "From the entrance, walk forward into the lobby."),

Structure:
- origin: Defines the node which the user is coming from
- target: Defines the point which the user will go to when following this edge
  - Can be terminal (the destination has been reached) or non-terminal (just a step on the way to the destination)
- weight: Distance between the nodes
  - Manhattan/rectilinear in this dataset, but could be defined as Euclidean distance
- description: Describes, in human terms, how to get from the origin to the target
  - Descriptions are chained together once shortest path from origin to destination is defined and passed back to Whisper for text-to-speech output

In this dataset, all nodes are defined on a single 2D plane. Stairs and lifts are treated as special cases, with higher weights between the nodes representing the stairs/lifts between floors than any other edge on the same floor, in order to prevent these edges from being preferred when they are not strictly necessary to reach the destination.

The following images illustrate how a real-world example, in this case, the second floor of Arás an Phiarsaigh, could be mapped to nodes and edges in a structure compatible with PERCI. The second floor corresponds to the right-hand side of the map of nodes and edges.

![IMG-20260325-WA0002](https://github.com/user-attachments/assets/199cb2c8-b302-4435-9015-3436e78b0370)

<img width="2007" height="1625" alt="node_map" src="https://github.com/user-attachments/assets/720d0748-02b6-4e5b-b7b4-7fe25c413a98" />

In addition to the names and IDs assigned to the nodes, a set of aliases may be defined for a given node in order to disambiguate requests for a specific destination. These are contained in a Dict object named DESTINATION_ALIASES, with the elements of the Dict being a string representing the node ID for a specific node and a List of strings comprising the aliases.

Example destination alias:

    DESTINATION_ALIASES: Dict[str, List[str]] = {
        [...]
        "cafe": ["cafe", "coffee area", "coffee shop"],
        [...]
    }

These destination aliases are passed to the Qwen LLM along with the node IDs in the user prompt.
