import sys
import json
import math
import heapq
import re
from typing import Dict, List, Tuple, Optional, Any

from llama_cpp import Llama

## LLM Model: Loaded once at startup, then reused for each request
# Model being used: Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
# Quantised for 4-bit integers using GGUF, run through llama.cpp
#
# Command for downloading dependencies (run before first starting):
#   pip install llama-cpp-python huggingface_hub
#   huggingface-cli download bartowski/Qwen2.5-0.5B-Instruct-GGUF \
#     --include "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf" --local-dir ./models

MODEL_PATH = "./models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"

print("[LLM] Loading model...", file=sys.stderr, flush=True)
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,      # was 512
    n_threads=4,
    n_gpu_layers=0,
    verbose=False,
)
print("[LLM] Model ready", file=sys.stderr, flush=True)

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
    "seminar_room": Node("seminar_room", "Seminar Room", 20, 14),
        
    # Easter egg
    "francois": Node("francois", "François's office", 50, -50),
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
    
    # Easter egg
    ("entrance", "francois", 100, "François is an enigma of the mind and does not want to be found."),
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
    "seminar_room": ["seminar"],
            
    # Easter egg
    "francois": ["francois", "françois"],
}

TOILETS = ["toilet"]

# Define a set of terms to be used to distinguish accessibility requests
ACCESSIBLE_KEYWORDS = [
    "accessible", "wheelchair", "step-free", "no stairs", "disabled"
]

## LLM intent parsing
# Define system prompt and functions for generating the response.

INTENT_SYSTEM_PROMPT = """You are a building wayfinding assistant. Your only job is to extract the user's intended destination from their message and return a JSON object.

Rules:
- Output ONLY valid JSON, no markdown, no explanation.
- If the user is asking to go somewhere, set intent to "navigation" and pick the best matching destination_id from the allowed list.
- If the message is not about navigation, set intent to "not_navigation" and destination_id to null.
- If you cannot identify the destination, set needs_clarification to true and destination_id to null.
- Only use destination_id values from the allowed list exactly as written.

JSON format:
{"intent": "navigation" | "not_navigation", "destination_id": string | null, "needs_clarification": boolean}"""

def generate_response(system_prompt: str, user_prompt: str, max_tokens: int = 80) -> str:
    """Generate JSON response by extracting information from the user prompt."""
    # Uses the ChatML format that Qwen instruct models expect
    result = llm.create_chat_completion(
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens = max_tokens,
        temperature = 0.0,  # Set system to be deterministic for reliable JSON
        stop = ["}\n", "\n\n"],
    )
    return result["choices"][0]["message"]["content"].strip()

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract generated JSON from the LLM response."""
    text = text.strip() # Remove leading/trailing whitespace
    try:
        obj = json.loads(text) # Decode JSON to a Python object
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Perform a regex search for opening/closing curly brackets
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0)) # Decode JSON to a Python object
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None

def fallback_intent_parser(question: str) -> Dict[str, Any]:
    """Pure string-matching fallback if the LLM produces bad output."""
    # Convert text to lower case and remove leading/trailing whitespace
    t = normalise(question) 

    navigation_phrases = [
        "where is", "how do i get", "take me", "directions", "nearest",
        "go to", "route", "way to", "find", "i need", "i want to go",
        "can you take me", "toilet", "bathroom", "restroom"
    ]

    # Iterate over the list of destination aliases and attempt to match them
    # in the user request
    destination_id = None
    for node_id, aliases in DESTINATION_ALIASES.items():
        for alias in aliases:
            if alias in t:
                destination_id = node_id
                break
        if destination_id is not None:
            break

    # If a destination ID is located, or one of the navigation phrases is found,
    # define the request as a navigation request
    is_nav = destination_id is not None or any(p in t for p in navigation_phrases)

    return {
        "intent": "navigation" if is_nav else "not_navigation",
        "destination_id": destination_id,
        "needs_clarification": False,
    }

def parse_user_request(question: str) -> Dict[str, Any]:
    '''Run user request through LLM or fallback parser, extracting information from the generated JSON object.'''
    # Generate list of valid destinations from the node IDs and names
    valid_destinations = build_valid_destination_text()

    # Concatenate transcribed user input and destination list
    user_prompt = f"""User message: {question}

    Allowed destination_id values:
    {valid_destinations}

    Return JSON only."""

    # Generate LLM response using system and user prompts
    raw = generate_response(
        system_prompt=INTENT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=80,
    )

    # Attempt to extract JSON object from the LLM response
    parsed = extract_json_object(raw) # JSON object converted to Python dict
    # Use the fallback parser if the LLM does not generate a valid response
    if parsed is None:
        print(f"[LLM] Bad output, using fallback. Raw: {raw!r}", file=sys.stderr)
        return fallback_intent_parser(question)

    # Extract relevant values from key-value pairs
    intent = parsed.get("intent")
    destination_id = parsed.get("destination_id")
    needs_clarification = bool(parsed.get("needs_clarification", False))

    # Use the fallback parser if the LLM does not generate a valid intent
    if intent not in {"navigation", "not_navigation"}:
        return fallback_intent_parser(question)

    # Reject any destination_id the LLM invented that isn't in our list
    if destination_id is not None and destination_id not in DESTINATION_ALIASES:
        print(f"[LLM] Unknown destination_id '{destination_id}', falling back.", file=sys.stderr)
        return fallback_intent_parser(question)

    return {
        "intent": intent,
        "destination_id": destination_id,
        "needs_clarification": needs_clarification,
    }

## Helper and navigation functions
def normalise(text: str) -> str:
    '''Convert text to lower case; remove leading/trailing whitespace'''
    return re.sub(r"\s+", " ", text.strip().lower())

def heuristic(a: str, b: str) -> float:
    '''Generate A* heuristic for distance remaining to destination'''
    na, nb = Node_List[a], Node_List[b]
    # Return Euclidean distance to destination
    return math.hypot(na.x - nb.x, na.y - nb.y)

def astar(start: str, goal: str) -> Optional[Tuple[List[str], float, List[str]]]:
    '''A*: Find shortest path to goal from start'''
    # Return if either start or goal is not in the list of nodes
    if start not in Node_List or goal not in Node_List:
        return None

    # Min-priority heap queue for nodes yet to be checked
    # Contains heuristic metric, current path length and node name
    open_heap: List[Tuple[float, float, str]] = []
    # Push start node to heap
    heapq.heappush(open_heap, (heuristic(start, goal), 0.0, start))
    
    came_from: Dict[str, Optional[str]] = {start: None}
    came_instr: Dict[str, Optional[str]] = {start: None}
    g_score: Dict[str, float] = {start: 0.0}
    visited: set = set()

    # Continue while there are still open nodes in the heap
    while open_heap:
        # Pop the node with the minimum heuristic value
        _, current_g, current = heapq.heappop(open_heap)
        if current in visited:
            continue # Ignore nodes that have already been visited
        visited.add(current) # Otherwise, add node to the list of visited nodes

        # If the current node is the goal, generate full path using backtracking
        if current == goal:
            steps: List[str] = []
            node_path: List[str] = []
            cur = goal
            while cur is not None:
                node_path.append(cur)
                if cur != start:
                    instr = came_instr[cur]
                    if instr is not None:
                        steps.append(instr)
                cur = came_from[cur]
            node_path.reverse()
            steps.reverse()
            return steps, g_score[goal], node_path

        # Check all of the nodes reachable from the current node.
        # Calculate the heuristic metric, then update the path if there is
        # a lower cost than the current path to that node.
        for neighbor, edge_cost, instr in Graph.get(current, []):
            # Calculate expected g(n) score as a combination of the current g(n)
            # score and the cost of the edge
            tentative_g = g_score[current] + edge_cost
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                came_instr[neighbor] = instr
                g_score[neighbor] = tentative_g
                heapq.heappush(
                    open_heap,
                    (tentative_g + heuristic(neighbor, goal), tentative_g, neighbor)
                )

    return None

def find_best_destination(start: str, destinations: List[str]) -> Tuple[Optional[str], Optional[List[str]], float, Optional[List[str]]]:
    '''Find the shortest path to one of a set of destinations of a certain type.'''
    best_dest = None
    best_steps = None
    best_cost = float("inf")
    best_node_path = None

    # Iteratively check each destination in the list and update if another path
    # is shorter than the current shortest path
    for dest in destinations:
        result = astar(start, dest)
        if result is None:
            continue
        steps, cost, node_path = result
        if cost < best_cost:
            best_dest = dest
            best_steps = steps
            best_cost = cost
            best_node_path = node_path

    return best_dest, best_steps, best_cost, best_node_path


def wants_accessible(text: str) -> bool:
    '''Check if the user input contains a request for accessibility.'''
    return any(k in normalise(text) for k in ACCESSIBLE_KEYWORDS)

def extract_start(text: str) -> str:
    '''Locate the start node for navigation.'''
    # Always starting from entrance since PERCI is at the building entrance
    return "entrance"

def build_valid_destination_text() -> str:
    '''Create list of node names and aliases to be passed to the LLM.'''
    lines = []
    for node_id, aliases in DESTINATION_ALIASES.items():
        node_name = Node_List[node_id].name
        lines.append(f"- {node_id}: {node_name} (e.g. {', '.join(aliases[:3])})")
    return "\n".join(lines)

## Route verbalisation
def verbalise_route(destination_name: str, steps: List[str]) -> str:
    '''Return the concatenated list of steps to reach a destination.'''
    if not steps:
        return f"I couldn't find a route to {destination_name}."
    # Remove leading/trailing whitespace
    clean_steps = [s.strip() for s in steps if s and s.strip()]
    return f"Directions to {destination_name}: {' '.join(clean_steps)}"


def answer(question: str) -> str:
    '''Generate an answer to be passed to Whisper and read to the user.'''
    parsed = parse_user_request(question)

    # If the LLM or fallback parser does not interpret this as a request for
    # directions, inform user
    if parsed["intent"] != "navigation":
        return "I can only help with directions inside this building. Please tell me where you would like to go."

    # If the LLM or fallback parser is unclear on destination, inform user
    if parsed["needs_clarification"]:
        return "I'm not sure which room you mean. Could you give me a bit more detail about where you'd like to go?"

    # Get the start and destination nodes from the parser
    start = extract_start(question)
    requested_destination = parsed["destination_id"]

    # Find shortest path using A*
    if requested_destination is not None:
        result = astar(start, requested_destination)
        if result is None:
            return "I'm unsure how to reach that destination. Please ask at reception for assistance."
        steps, cost, node_path = result
        return verbalise_route(Node_List[requested_destination].name, steps)

    # No specific destination — check if they want a toilet generically
    t = normalise(question)
    if any(w in t for w in ["toilet", "bathroom", "restroom", "loo", "gents", "ladies"]):
        best_toilet, steps, cost, node_path = find_best_destination(start, TOILETS)
        if not best_toilet or not steps:
            return "I'm unsure how to reach the toilets. Please ask at reception for assistance."
        return verbalise_route(Node_List[best_toilet].name, steps)

    return "I can only help with directions inside this building. Please tell me where you would like to go."

## MAIN LOOP: Reads JSON lines from stdin, writes JSON lines to stdout

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            user_text = req.get("text", "")
            response = answer(user_text)
            print(json.dumps({"response": response}), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
