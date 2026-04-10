from typing import Dict, List, Tuple, Optional, Any

# This may be used as a template for definitions of a dataset for PERCI.
# Place the modified data in place of the respective parts of llm_worker3.py.

## Building data: Nodes and edges
# Nodes have ID, human-readable name, x/y coordinates
# Edges have origin & destination nodes, weight, description text
# Edge weight defined as Manhattan distance between origin & destination

class Node:
    def __init__(self, node_id: str, name: str, x: float, y: float):
        self.id = node_id
        self.name = name
        self.x = x
        self.y = y

Node_List: Dict[str, Node] = {
    # Ground floor intermediate points
    "doors_1": Node("doors_1", "doors to the ground floor lobby", -2, 6),
    "doors_2": Node("doors_2", "doors to the stairs", -2, 10),
    "lobby": Node("lobby", "lobby", -2, 0),
    
    "stairs_gf": Node("stairs_gf", "the stairs", -2, 12),
    "lift_gf": Node("lift_gf", "the lift", 0, 12),
    
    # Ground floor destinations
    "entrance": Node("entrance", "entrance", 0, 0),
    "reception": Node("reception", "Reception", -4, 2),
    "cafe": Node("cafe", "café", -8, 0),
    "it": Node("it", "Service Desk", -1, 4),
    "0.09": Node("0.09", "Lecture Room 0.09", -4, 6),
    "0.12": Node("0.12", "AAP 0.12", -6, 6),
    "0.13": Node("0.13", "Server Room 0.13", -6, 8),
    
    "toilet": Node("toilet", "toilets", 0, 10),
        
    # 2nd floor - add 20 to all horizontal X coordinates
    # 2nd floor intermediate points   
    "stairs_2f": Node("stairs_2f", "the stairs", 19, 12),
    "lift_2f": Node("lift_2f", "the lift", 20, 12),
    
    # Define the corridor corners in clockwise order
    #"top_stairs_2f": Node("top_stairs_2f", "the stairs area", 19, 12),
    "se_corner": Node("se_corner", "the south-east corridor corner", 19, 1),
    "sw_corner": Node("sw_corner", "the south-west corridor corner", 12, 1),
    "nw_corner": Node("nw_corner", "the north-west corridor corner", 12, 14),
    "ne_corner": Node("ne_corner", "the north-east corridor corner", 19, 14),
    
    # 2nd floor destinations
    "common_room": Node("2.16", "Common Room 2.16", 23, 1),
    "meeting_room": Node("2.17", "Meeting Room 2.17", 22, 0),
    "2.18": Node("2.18", "Hot Desk 2.18", 21, 0),
    "2.19": Node("2.19", "Alan Smithee 2.19", 20, 0),
    "2.20": Node("2.20", "George Sand 2.20", 19, 0),
    "lab": Node("lab", "Undergraduate Lab 2.15", 20, 13),
    "mail_room": Node("2.22", "Mail Room 2.22", 17, 0),
    "2.23": Node("2.23", "Cordwainer Bird 2.23", 16, 0),
    "2.24": Node("2.24", "Eugene O'Rourke 2.24", 14, 0),
    "cadlab": Node("cadlab", "Cad Lab 2.28", 12, 0),
    "2.25": Node("2.25", "Senior Room 2.25", 13, 2),
    "2.26": Node("2.26", "Paul Merchant 2.26", 13, 8),
    "2.27": Node("2.27", "Walter Plinge 2.27", 13, 9),
    "project_space": Node("project_space", "Project Space 2.31", 13, 13),
    "workshop": Node("workshop", "Workshop 2.30", 11, 13),
    "comp_lab": Node("comp_lab", "Computer Lab 2.02", 11, 15),
    "2.03": Node("2.03", "Lecture Room 2.03", 19, 15),
    "2.04": Node("2.04", "Lecture Room 2.04", 20, 15),
    "seminar_room": Node("seminar_room", "Seminar Room", 20, 14)
}

Edge_List = [
    # Ground floor
    ("entrance", "lobby", 2, "From the entrance, walk forward into the lobby."),
    ("entrance", "cafe", 8, "From the entrance, the cafe is directly in front of you."),
    
    ("lobby", "reception", 4, "The reception will be on your right-hand side."),
    ("lobby", "it", 5, "Turning to the right, the service desk will be to your right down the corridor."),
    ("lobby", "doors_1", 8, "Turning to the right, pass through the doors in front of you."),
    
    ("doors_1", "0.09", 2, "Turning left, Lecture Room 0.09 will be on the left."),
    ("doors_1", "0.12", 4, "Turning left, AAP 0.12 will be in front of you."),
    ("doors_1", "0.13", 6, "Turning left, server room 0.13 will be in front of you, to the right of AAP 0.12."),
    ("doors_1", "doors_2", 4, "Continue forward through the doors in front of you."),

    ("doors_2", "toilet", 1, "Turning to the right, the toilets are on the right."),
    ("doors_2", "stairs_gf", 2, "Take a dogleg to the right to the stairs."),
    ("doors_2", "lift_gf", 4, "Turning to the right, the lift will be beside the disabled toilets."),
    
    # Stairs/lift between ground floor and 2nd floor
    ("stairs_gf", "stairs_2f", 20, "Go up the stairs to the 2nd floor."),
    ("lift_gf", "lift_2f", 20, "Go up the lift to the 2nd floor."),
    
    # 2nd floor
    ("stairs_2f", "se_corner", 12, "Continue through the two sets of doors in front of you."),
    ("stairs_2f", "ne_corner", 3, "Turn to the left, then continue through the doors to your left."),
    
    ("lift_2f", "se_corner", 12, "Continue through the two sets of doors located to your left."),
    ("lift_2f", "ne_corner", 3, "Turn to the left, then continue through the doors to your left."),
    
    # Corner-to-corner, clockwise:
    ("se_corner", "sw_corner", 7, "Turn to the right, then continue through the two sets of doors in front of you."),
    ("sw_corner", "nw_corner", 13, "Turn to the right, then continue down the corridor, through the door."),
    ("nw_corner", "ne_corner", 7, "Turn to the right, then continue through both sets of doors ahead."),
    ("ne_corner", "stairs_2f", 3, "Turn right, then continue through the stairs to the top of the stairs."),
    
    # Corner-to-corner, anti-clockwise:
    ("ne_corner", "nw_corner", 7, "Turn to the left, then continue through the two sets of doors ahead."),
    ("nw_corner", "sw_corner", 13, "Turn to the left, then continue down the corridor through the door."),
    ("sw_corner", "se_corner", 7, "Turn to the left, then continue through both sets of doors in front of you."),
    ("se_corner", "stairs_2f", 12, "Turn towards the two sets of doors and continue toward the stairs."),
    
    # 2nd floor NE corner points
    ("ne_corner", "lab", 2, "The undergraduate lab will be on the right."),
    ("ne_corner", "seminar_room", 1, "The seminar room will be on the right."),
    ("ne_corner", "2.04", 2, "Lecture Room 2.04 will be ahead to the right."),
    ("ne_corner", "2.03", 1, "Lecture Room 2.03 will be ahead on the left."),
    
    # 2nd floor NW corner points
    ("nw_corner", "comp_lab", 2, "The computer lab will be in front of you."),
    ("nw_corner", "workshop", 2, "Turn to the left. The workshop will be on the right around the corner."),
    ("nw_corner", "project_space", 2, "Turn to the left. The project space will be on the left around the corner."),
    ("nw_corner", "2.27", 6, "Turn to the left. Continue through the door. Office 2.27 will be on your left."),
    ("nw_corner", "2.26", 7, "Turn to the left. Continue through the door. Office 2.26 will be on your left past 2.27."),
    
    # 2nd floor SW corner points
    ("sw_corner", "2.24", 3, "Office 2.24 will be on your left immediately past the second door."),
    ("sw_corner", "cadlab", 1, "The Cad Lab will be on your left past office 2.24."),
    ("sw_corner", "2.25", 2, "Turn to the right. Office 2.25 will be on your right around the corner."),
    
    # 2nd floor SE corner points
    ("se_corner", "2.23", 4, "Turn to the right and go through the door. Office 2.23 will be on your left past the mail room."),
    ("se_corner", "mail_room", 3, "Turn to the right, continuing through the door. The mail room will be on your left."),
    ("se_corner", "2.20", 1, "Office 2.20 will be directly in front of you."),
    ("se_corner", "2.19", 2, "Turn to the left. Office 2.19 will be on the right past office 2.20."),
    ("se_corner", "2.18", 3, "Turn to the left. The hot desk office, 2.18, will be on the right down the corridor."),
    ("se_corner", "meeting_room", 4, "Turn to the left. The meeting room will be on the right at the end of the corridor."),
    ("se_corner", "common_room", 4, "Turn to the left. The common room is directly ahead at the end of the corridor."),
]

# Define the graph of nodes, connected by edges
Graph: Dict[str, List[Tuple[str, float, str]]] = {nid: [] for nid in Node_List}
for a, b, cost, instr in Edge_List:
    Graph[a].append((b, cost, instr))
    Graph[b].append((a, cost, f"Go back toward {Node_List[a].name}.")) # Simple reverse node traversal

## Building data: Destination aliases
# Define a set of common aliases for various destination nodes
# This helps disambiguate some requests that would otherwise be misinterpreted

DESTINATION_ALIASES: Dict[str, List[str]] = {
    # Ground floor destinations
    "reception": ["front desk"],
    "cafe": ["cafe", "coffee area", "coffee shop"],
    "it": ["it"],
    "0.09": ["0.09"],
    "0.12": ["0.12", "aap 2", "aap"],
    "0.13": ["0.13", "server room"],

    "toilet": ["toilet", "lavatory", "restroom", "bathroom", "loo"],
        
    # 2nd floor destinations
    "common_room": ["common room", "2.16"],
    "meeting_room": ["meeting room", "2.17"],
    "2.18": ["2.18", "hot desk"],
    "2.19": ["smithee", "alan", "2.19"],
    "2.20": ["sand", "george", "2.20"],
    "mail_room": ["mail room", "2.22"],
    "2.23": ["bird", "cordwainer", "2.23"],
    "2.24": ["o'rourke", "eugene", "2.24",],
    "cadlab": ["cadlab", "cad lab", "card lab", "cat lab", "2.28"],
    "lab": ["lab", "undergraduate lab", "2.15"],
    "2.25": ["2.25", "senior room"],
    "2.26": ["merchant", "paul", "2.26"],
    "2.27": ["plinge", "walter", "2.27"],
    "project_space": ["project space", "2.31"],
    "workshop": ["workshop", "2.30"],
    "comp_lab": ["computer", "computer lab", "comp lab", "2.02"],
    "2.03": ["2.03"],
    "2.04": ["2.04"],
    "seminar_room": ["seminar"]
}
