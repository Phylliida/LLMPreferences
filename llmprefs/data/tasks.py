import os
import json

def loadNeutralTasks():
    taskDir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(taskDir, "neutral.csv"), newline="", encoding="utf-8") as f:
        neutralTasks = [task.strip() for task in f.read().split("\n") if len(task.strip()) > 0]
    return neutralTasks

def loadWildchatTasks():
    taskDir = os.path.dirname(os.path.realpath(__file__))
    # Tasks from Daniel Paleka
    with open(os.path.join(taskDir, "wildchat_1880.jsonl"), newline="", encoding="utf-8") as f:
        lines = [line.strip() for line in f.read().split("\n") if len(line.strip()) > 0]
        return [json.loads(line)['text'] for line in lines]

def loadTasks():
   tasks = loadNeutralTasks() + loadWildchatTasks()