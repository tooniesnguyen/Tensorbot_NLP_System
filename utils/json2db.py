import pymongo
import json
from pymongo import MongoClient, InsertOne

connect_str = "mongodb://Nhan:1@127.0.0.1:27017/tensorbot_nlp"
json_dir = "/home/toonies/Learn/AI_CKM/Tensorbot_NLP_System/data/dicts/intents.json"
client = MongoClient(connect_str)
db = client.tensorbot_nlp
collection = db["dialog_nlp"]
requesting = []

with open(json_dir, 'r') as f:
    data = json.load(f)
    for intent in data["intents"]:
        requesting.append(InsertOne(intent))
        
collection.bulk_write(requesting)

print("Update successful")
