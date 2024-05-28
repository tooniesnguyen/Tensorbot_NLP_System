import pymongo
import json
import random
from pymongo import MongoClient

connect_str = "mongodb://Nhan:1@127.0.0.1:27017/tensorbot_nlp"
client = MongoClient(connect_str)
db = client.tensorbot_nlp
collection = db["dialog_nlp"]

def get_random_response(tag_name):
    document = collection.find_one({"tag": tag_name}, {'_id': 0, 'responses': 1})
    if document:
        responses = document['responses']
        return random.choice(responses)
    else:
        return None
    client.close()

def main():
    tag_name = "thanks"
    random_response = get_random_response(tag_name)
    if random_response:
        print(f"Random response for tag [{tag_name}]: {random_response}")
    else:
        print(f"No responses found for tag [{tag_name}]")

if __name__ == "__main__":
    main()
