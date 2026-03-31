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
    
    #"toilets_f_gf": Node("toilets_f_gf", "women's toilets", -1, 10),
    #"toilets_m_gf": Node("toilets_m_gf", "men's toilets", 0, 10),
    #"toilets_disabled": Node("toilets_disabled", "disabled toilets", 0, 11),
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
    #"toilets_m_2f": Node("toilets_m_2f", "men's toilets", 20, 11),
    "common_room": Node("2.16", "Common Room 2.16", 23, 1),
    "meeting_room": Node("2.17", "Meeting Room 2.17", 22, 0),
    "2.18": Node("2.18", "Hot Desk 2.18", 21, 0),
    "2.19": Node("2.19", "Office 2.19", 20, 0),
    "2.20": Node("2.20", "Office 2.20", 19, 0),
    "lab": Node("2.15", "Undergraduate Lab 2.15", 20, 13),
    "mail_room": Node("2.22", "Mail Room 2.22", 17, 0),
    "2.23": Node("2.23", "Office 2.23", 16, 0),
    "2.24": Node("2.24", "Eugene O'Rourke 2.24", 14, 0),
    "cadlab": Node("cadlab", "Cad Lab 2.28", 12, 0),
    "2.25": Node("2.25", "Senior Room 2.25", 13, 2),
    "2.26": Node("2.26", "Office 2.26", 13, 8),
    "2.27": Node("2.27", "Office 2.27", 13, 9),
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
    #("doors_2", "toilets_f_gf", 1, "Turning to the right, the women's toilets are on the right."),
    #("doors_2", "toilets_m_gf", 2, "Turning to the right, the men's toilets are on the right, past the women's toilets."),
    #("doors_2", "toilets_disabled", 3, "Turning to the right, the disabled toilets are in front of you."),
    ("doors_2", "stairs_gf", 2, "Take a dogleg to the right to the stairs."),
    ("doors_2", "lift_gf", 4, "Turning to the right, the lift will be beside the disabled toilets."),
    
    # Stairs/lift between ground floor and 2nd floor
    ("stairs_gf", "stairs_2f", 20, "Go up the stairs to the 2nd floor."),
    ("lift_gf", "lift_2f", 20, "Go up the lift to the 2nd floor."),
    
    # 2nd floor
    #("stairs_2f", "toilets_m_2f", 3, "The men's toilets will be on the left."),
    ("stairs_2f", "se_corner", 12, "Continue through the two sets of doors in front of you."),
    ("stairs_2f", "ne_corner", 3, "Turn to the left, then continue through the doors to your left."),
    
    #("lift_2f", "toilets_m_2f", 1, "The men's toilets will be on your left."),
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
    ("entrance", "francois", 100, "François's office is an enigma and does not want to be found."),
]

# Original node list
'''Node_List: Dict[str, Node] = {
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
}'''

'''Edge_List = [
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
]'''

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

'''DESTINATION_ALIASES: Dict[str, List[str]] = {
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
}'''

DESTINATION_ALIASES: Dict[str, List[str]] = {
    # Ground floor intermediate points
    "doors_1": ["doors to the ground floor lobby"],
    "doors_2": ["doors to the stairs"],
    "lobby": ["lobby"],
        
    "stairs_gf": ["ground floor stairs"],
    "lift_gf": ["the ground floor lift"],
    
    # Ground floor destinations
    "entrance": ["entrance", "exit", "front entrance"],
    "reception": ["reception", "front desk"],
    "cafe": ["cafe", "coffee area", "coffee shop"],
    "it": ["service desk", "it"],
    "0.09": ["lecture room 0.09"],
    "0.12": ["0.12", "aap 0.12", "aap"],
    "0.13": ["0.13", "server room 0.13", "server room"],

    "toilet": ["toilet", "toilets"],
    #"toilets_m_gf": ["men's toilet", "mens toilet", "male toilet", "gents toilet", "men's bathroom", "mens bathroom", "gents"],
    #"toilets_f_gf": ["women's toilet", "womens toilet", "female toilet", "ladies toilet", "women's bathroom", "womens bathroom", "ladies"],
    #"toilets_disabled": ["disabled toilet", "disabled bathroom", "disabled restroom"],
        
    # 2nd floor - add 20 to all horizontal X coordinates
    # 2nd floor intermediate points   
    "stairs_2f": ["the second-floor stairs"],
    "lift_2f": ["the second-floor lift"],
    
    # Define the corridor corners in clockwise order
    "se_corner": ["the south-east corridor corner"],
    "sw_corner": ["the south-west corridor corner"],
    "nw_corner": ["the north-west corridor corner"],
    "ne_corner": ["the north-east corridor corner"],
    
    # 2nd floor destinations
    #"toilets_m_2f": ["men's toilet", "mens toilet", "male toilet", "gents toilet", "men's bathroom", "mens bathroom", "gents"],
    "common_room": ["2.16", "common room", "common room 2.16"],
    "meeting_room": ["Meeting Room 2.17"],
    "2.18": ["hot desk 2.18"],
    "2.19": ["2.19"],
    "2.20": ["2.20"],
    "lab": ["undergraduate lab 2.15"],
    "mail_room": ["mail room 2.22"],
    "2.23": ["2.23"],
    "2.24": ["eugene", "o'rourke", "2.24", "eugene o'rourke 2.24"],
    "cadlab": ["cadlab", "2.28"],
    "2.25": ["2.25", "senior room", "senior room 2.25"],
    "2.26": ["2.26"],
    "2.27": ["2.27"],
    "project_space": ["project space", "2.31", "project space 2.31"],
    "workshop": ["workshop", "2.30", "workshop 2.30"],
    "comp_lab": ["computer", "computer lab", "comp lab", "2.02", "computer lab 2.02"],
    "2.03": ["2.03", "lecture room 2.03"],
    "2.04": ["2.04", "lecture room 2.04"],
    "seminar_room": ["seminar room", "seminar"],
            
    # Easter egg
    "francois": ["francois", "françois", "françois's office"],
}

#TOILETS = ["toilets_m_gf", "toilets_f_gf", "toilets_m_2f"]
TOILETS = ["toilet"]

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
