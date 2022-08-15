import json


def save_json(obj,path):
    with open(path,'w') as f:
        json.dump(obj,f,indent=4)

def load_json(path):
    with open(path,'r') as f:
        res = json.load(f)
    return res