# Just an example Richard and Eimhin to replace with real file when finished

import sys
import json
import math
import heapq
import re
from typing import Dict, List, Tuple, Optional, Any

from llama_cpp import Llama


class Node:
    def __init__(self, node_id: str, name: str, x: float, y: float):
        self.id = node_id
        self.name = name
        self.x = x
        self.y = y


Node_List: Dict[str, Node] = {
    "entrance": Node("entrance", "Entrance", 0, 0),
    "lobby": Node("lobby", "Lobby", 0, 4),
    "reception": Node("reception", "Reception", 1, 4),
    "cafe": Node("cafe", "Cafe", -3, 4),
    "fountain": Node("fountain", "Water Fountain", -1, 6),
    "doors": Node("doors", "Past the Doors", 0, 9),
    "toilet_m": Node("toilet_m", "Men's Toilets", -3, 8),
    "toilet_f": Node("toilet_f", "Women's Toilets", 3, 8),
    "meeting_room": Node("meeting_room", "Meeting Room 0.01", 6, 8),
    "lecture_room_1": Node("lecture_room_1", "Lecture Room 0.02", 7, 9),
    "supply_closet_1": Node("supply_closet_1", "Supply Closet 1", 7, 11),
    "lecture_room_2": Node("lecture_room_2", "Lecture Room 0.03", 7, 13),
    "lecture_room_3": Node("lecture_room_3", "Large Lecture Hall 0.04", 7, 15),
    "canteen": Node("canteen", "Staff Canteen 0.05", 7, 21),
    "lab": Node("lab", "Student Lab 0.06", 6, 22),
    "lecture_room_4": Node("lecture_room_4", "Lecture Room 0.07", 2, 22),
    "computer_room": Node("computer_room", "Computer Room 0.08", -2, 22),
    "smithee": Node("smithee", "Alan Smithee 0.09", -4, 22),
    "sand": Node("sand", "George Sand 0.10", -6, 22),
    "bird": Node("bird", "Cordwainer Bird 0.11", -7, 21),
    "merchant": Node("merchant", "Paul Merchant 0.12", -7, 19),
    "lecture_room_5": Node("lecture_room_5", "Lecture Room 0.13", -7, 17),
    "supply_closet_2": Node("supply_closet_2", "Supply Closet 2", -7, 15),
    "plinge": Node("plinge", "Walter Plinge 0.14", -7, 13),
    "agnew": Node("agnew", "David Agnew 0.15", -7, 11),
    "jaynes": Node("jaynes", "Roderick Jaynes 0.16", -6, 8),
    "sw_corridor": Node("sw_corridor", "South-West Corridor Corner", -6, 9),
    "nw_corridor": Node("nw_corridor", "North-West Corridor Corner", -6, 21),
    "ne_corridor": Node("ne_corridor", "North-East Corridor Corner", 6, 21),
    "se_corridor": Node("se_corridor", "South-East Corridor Corner", 6, 9),
}

Edge_List = [
    ("entrance", "lobby", 4, "From the entrance, walk forward into the lobby."),
    ("lobby", "entrance", 4, "With the reception on your left, the exit will be in front of you."),
    ("lobby", "reception", 1, "The reception will be on the right."),
    ("lobby", "cafe", 3, "The cafe will be on the left."),
    ("lobby", "fountain", 3, "The water fountain will be on the left corner ahead."),
    ("lobby", "doors", 4, "Go through the doors to your front."),
    ("doors", "toilet_m", 4, "Turn to the left. The men's toilets will be on your left down the corridor."),
    ("doors", "jaynes", 7, "Turn to the left. Doctor Jaynes' office is at the end of the corridor on the left."),
    ("doors", "sw_corridor", 6, "Go down the corridor and turn to the right."),
    ("sw_corridor", "agnew", 3, "Doctor Agnew's office will be on the left."),
    ("sw_corridor", "plinge", 5, "Down the corridor, Doctor Plinge's office is on the left, past Doctor Agnew's office."),
    ("sw_corridor", "supply_closet_2", 7, "Walk down the corridor. Supply Closet 2 is on the left, past Doctor Plinge's office."),
    ("sw_corridor", "lecture_room_5", 9, "Lecture Room 0.13 is down the corridor on the left, past Supply Closet 2."),
    ("sw_corridor", "merchant", 10, "Professor Merchant's office is down the corridor on the left, past Lecture Room 0.13."),
    ("sw_corridor", "bird", 13, "Doctor Bird's office is at the end of the corridor on the left."),
    ("sw_corridor", "sand", 13, "Doctor Sand's office is at the end of the corridor, straight ahead."),
    ("sw_corridor", "nw_corridor", 12, "Go down the corridor and turn right."),
    ("nw_corridor", "smithee", 3, "Professor Smithee's office is on the left, past Doctor Sand's office."),
    ("nw_corridor", "computer_room", 5, "Computer Room 0.08 is down the corridor on the left, past Professor Smithee's office."),
    ("nw_corridor", "lecture_room_4", 9, "Lecture Room 0.07 is down the corridor on the left, past Computer Room 0.08."),
    ("nw_corridor", "lab", 13, "Student Lab 0.06 is at the end of the corridor on the left."),
    ("nw_corridor", "canteen", 13, "The staff canteen is at the end of the corridor, straight ahead."),
    ("nw_corridor", "ne_corridor", 12, "Go down the corridor and turn right."),
    ("ne_corridor", "lecture_room_3", 7, "Large Lecture Hall 0.04 is down the corridor on the left."),
    ("ne_corridor", "lecture_room_2", 9, "Lecture Room 0.03 is down the corridor on the left, past Large Lecture Hall 0.04."),
    ("ne_corridor", "supply_closet_1", 11, "Supply Closet 1 is down the corridor on the left, past Lecture Room 0.03."),
    ("ne_corridor", "lecture_room_1", 13, "Lecture Room 0.02 is at the end of the corridor on the left."),
    ("ne_corridor", "meeting_room", 13, "The meeting room is at the end of the corridor, straight ahead."),
    ("ne_corridor", "se_corridor", 12, "Go down the corridor and turn right."),
    ("se_corridor", "toilet_f", 4, "The women's toilets are to the left, past the meeting room."),
    ("se_corridor", "doors", 6, "Go down the corridor, towards the doors to the lobby."),
    ("doors", "toilet_f", 4, "Turn to the right. The women's toilets will be on your right down the corridor."),
    ("doors", "meeting_room", 7, "Turn to the right. The meeting room is at the end of the corridor on the right."),
    ("doors", "lecture_room_1", 7, "Turn to the right. Lecture Room 0.02 is at the end of the corridor, straight ahead."),
    ("doors", "se_corridor", 6, "Go down the corridor and turn left."),
    ("se_corridor", "supply_closet_1", 3, "Supply Closet 1 is on the right, past Lecture Room 0.02."),
    ("se_corridor", "lecture_room_2", 5, "Lecture Room 0.03 is down the corridor on the right, past Supply Closet 1."),
    ("se_corridor", "lecture_room_3", 7, "Large Lecture Hall 0.04 is down the corridor on the right, past Lecture Room 0.03."),
    ("se_corridor", "canteen", 13, "The staff canteen is at the end of the corridor on the right."),
    ("se_corridor", "lab", 13, "Student Lab 0.06 is at the end of the corridor, straight ahead."),
    ("se_corridor", "ne_corridor", 12, "Go down the corridor and turn left."),
    ("ne_corridor", "lecture_room_4", 5, "Lecture Room 0.07 is down the corridor on the left."),
    ("ne_corridor", "computer_room", 9, "Computer Room 0.08 is down the corridor on the right, past Lecture Room 0.07."),
    ("ne_corridor", "smithee", 11, "Professor Smithee's office is down the corridor on the right, past Computer Room 0.08."),
    ("ne_corridor", "sand", 13, "Doctor Sand's office is at the end of the corridor on the right."),
    ("ne_corridor", "bird", 13, "Doctor Bird's office is at the end of the corridor, straight ahead."),
    ("ne_corridor", "nw_corridor", 12, "Go down the corridor and turn left."),
    ("nw_corridor", "merchant", 3, "Professor Merchant's office is down the corridor on the right."),
    ("nw_corridor", "lecture_room_5", 5, "Lecture Room 0.13 is down the corridor on the right, past Professor Merchant's office."),
    ("nw_corridor", "supply_closet_2", 7, "Supply Closet 2 is down the corridor on the right, past Lecture Room 0.13."),
    ("nw_corridor", "plinge", 9, "Doctor Plinge's office is down the corridor on the right, past Supply Closet 2."),
    ("nw_corridor", "agnew", 11, "Doctor Agnew's office is down the corridor on the right, past Doctor Plinge's office."),
    ("nw_corridor", "jaynes", 13, "Doctor Jayne's office is at the end of the corridor, straight ahead."),
    ("nw_corridor", "sw_corridor", 12, "Go down the corridor and turn left."),
    ("sw_corridor", "toilet_m", 4, "The men's toilets are on the right, past Doctor Jaynes' office."),
    ("sw_corridor", "doors", 6, "Go down the corridor, towards the doors to the lobby."),
]

Graph: Dict[str, List[Tuple[str, float, str]]] = {nid: [] for nid in Node_List}
for a, b, cost, instr in Edge_List:
    Graph[a].append((b, cost, instr))
    Graph[b].append((a, cost, f"Go back toward {Node_List[a].name}."))


# ---------------------------------------------------------------------------
# Model — loaded ONCE at startup, reused for every request
# ---------------------------------------------------------------------------

# Download command (run once before starting):
#   pip install llama-cpp-python huggingface_hub
#   huggingface-cli download bartowski/Qwen2.5-1.5B-Instruct-GGUF \
#       --include "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf" --local-dir ./models

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


# ---------------------------------------------------------------------------
# Building data
# ---------------------------------------------------------------------------

DESTINATION_ALIASES: Dict[str, List[str]] = {
    "entrance": ["entrance", "exit", "front entrance"],
    "lobby": ["lobby"],
    "reception": ["reception", "front desk"],
    "cafe": ["cafe", "coffee area", "coffee shop"],
    "fountain": ["fountain", "water fountain"],
    "doors": ["doors", "past the doors"],
    "toilet_m": ["men's toilet", "mens toilet", "male toilet", "gents toilet", "men's bathroom", "mens bathroom", "gents"],
    "toilet_f": ["women's toilet", "womens toilet", "female toilet", "ladies toilet", "women's bathroom", "womens bathroom", "ladies"],
    "meeting_room": ["meeting room", "meeting room 0.01"],
    "lecture_room_1": ["lecture room 0.02", "room 0.02"],
    "supply_closet_1": ["supply closet 1"],
    "lecture_room_2": ["lecture room 0.03", "room 0.03"],
    "lecture_room_3": ["large lecture hall 0.04", "lecture hall 0.04", "room 0.04", "large lecture hall", "lecture hall"],
    "canteen": ["canteen", "staff canteen", "staff canteen 0.05"],
    "lab": ["lab", "student lab", "student lab 0.06"],
    "lecture_room_4": ["lecture room 0.07", "room 0.07"],
    "computer_room": ["computer room", "computer room 0.08", "computers"],
    "smithee": ["alan smithee", "professor smithee", "smithee"],
    "sand": ["george sand", "doctor sand", "sand"],
    "bird": ["cordwainer bird", "doctor bird", "bird"],
    "merchant": ["paul merchant", "professor merchant", "doctor merchant", "merchant"],
    "lecture_room_5": ["lecture room 0.13", "room 0.13"],
    "supply_closet_2": ["supply closet 2"],
    "plinge": ["walter plinge", "doctor plinge", "plinge"],
    "agnew": ["david agnew", "doctor agnew", "agnew"],
    "jaynes": ["roderick jaynes", "doctor jaynes", "jaynes"],
}

TOILETS = ["toilet_m", "toilet_f"]

ACCESSIBLE_KEYWORDS = [
    "accessible", "wheelchair", "step-free", "no stairs", "disabled access"
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def heuristic(a: str, b: str) -> float:
    na, nb = Node_List[a], Node_List[b]
    return math.hypot(na.x - nb.x, na.y - nb.y)


def astar(start: str, goal: str) -> Optional[Tuple[List[str], float, List[str]]]:
    if start not in Node_List or goal not in Node_List:
        return None

    open_heap: List[Tuple[float, float, str]] = []
    heapq.heappush(open_heap, (heuristic(start, goal), 0.0, start))

    came_from: Dict[str, Optional[str]] = {start: None}
    came_instr: Dict[str, Optional[str]] = {start: None}
    g_score: Dict[str, float] = {start: 0.0}
    visited: set = set()

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

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

        for neighbor, edge_cost, instr in Graph.get(current, []):
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
    best_dest = None
    best_steps = None
    best_cost = float("inf")
    best_node_path = None

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
    return any(k in normalize(text) for k in ACCESSIBLE_KEYWORDS)


def extract_start(text: str) -> str:
    # Always starting from entrance since PERCI is at the building entrance
    return "entrance"


def build_valid_destination_text() -> str:
    lines = []
    for node_id, aliases in DESTINATION_ALIASES.items():
        node_name = Node_List[node_id].name
        lines.append(f"- {node_id}: {node_name} (e.g. {', '.join(aliases[:3])})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM intent parsing
# ---------------------------------------------------------------------------

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
    # Uses the ChatML format that Qwen instruct models expect
    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,   # deterministic — important for JSON extraction
        stop=["}\n", "\n\n"],
    )
    return result["choices"][0]["message"]["content"].strip()


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def fallback_intent_parser(question: str) -> Dict[str, Any]:
    """Pure string-matching fallback if the LLM produces bad output."""
    t = normalize(question)

    navigation_phrases = [
        "where is", "how do i get", "take me", "directions", "nearest",
        "go to", "route", "way to", "find", "i need", "i want to go",
        "can you take me", "toilet", "bathroom", "restroom"
    ]

    destination_id = None
    for node_id, aliases in DESTINATION_ALIASES.items():
        for alias in aliases:
            if alias in t:
                destination_id = node_id
                break
        if destination_id is not None:
            break

    is_nav = destination_id is not None or any(p in t for p in navigation_phrases)

    return {
        "intent": "navigation" if is_nav else "not_navigation",
        "destination_id": destination_id,
        "needs_clarification": False,
    }


def parse_user_request(question: str) -> Dict[str, Any]:
    valid_destinations = build_valid_destination_text()

    user_prompt = f"""User message: {question}

Allowed destination_id values:
{valid_destinations}

Return JSON only."""

    raw = generate_response(
        system_prompt=INTENT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=80,
    )

    parsed = extract_json_object(raw)
    if parsed is None:
        print(f"[LLM] Bad output, using fallback. Raw: {raw!r}", file=sys.stderr)
        return fallback_intent_parser(question)

    intent = parsed.get("intent")
    destination_id = parsed.get("destination_id")
    needs_clarification = bool(parsed.get("needs_clarification", False))

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


# ---------------------------------------------------------------------------
# Route verbalization
# ---------------------------------------------------------------------------

def verbalize_route(destination_name: str, steps: List[str]) -> str:
    if not steps:
        return f"I couldn't find a route to {destination_name}."
    clean_steps = [s.strip() for s in steps if s and s.strip()]
    return f"Directions to {destination_name}: {' '.join(clean_steps)}"


def answer(question: str) -> str:
    parsed = parse_user_request(question)

    if parsed["intent"] != "navigation":
        return "I can only help with directions inside this building. Please tell me where you would like to go."

    if parsed["needs_clarification"]:
        return "I'm not sure which room you mean. Could you give me a bit more detail about where you'd like to go?"

    start = extract_start(question)
    requested_destination = parsed["destination_id"]

    if requested_destination is not None:
        result = astar(start, requested_destination)
        if result is None:
            return "I'm unsure how to reach that destination. Please ask at reception for assistance."
        steps, cost, node_path = result
        return verbalize_route(Node_List[requested_destination].name, steps)

    # No specific destination — check if they want a toilet generically
    t = normalize(question)
    if any(w in t for w in ["toilet", "bathroom", "restroom", "loo", "gents", "ladies"]):
        best_toilet, steps, cost, node_path = find_best_destination(start, TOILETS)
        if not best_toilet or not steps:
            return "I'm unsure how to reach the toilets. Please ask at reception for assistance."
        return verbalize_route(Node_List[best_toilet].name, steps)

    return "I can only help with directions inside this building. Please tell me where you would like to go."


# ---------------------------------------------------------------------------
# Main loop — reads JSON lines from stdin, writes JSON lines to stdout
# ---------------------------------------------------------------------------

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