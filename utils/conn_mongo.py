import pymongo
import json
import random
from pymongo import MongoClient

# Connection string to connect to the MongoDB database
connect_str = "mongodb://Nhan:1@127.0.0.1:27017/tensorbot_nlp"
# Create a MongoClient instance to connect to the MongoDB server
client = MongoClient(connect_str)
# Access the database named "tensorbot_nlp"
db = client.tensorbot_nlp
# Access the collection named "dialog_nlp" within the database
collection = db["dialog_nlp"]

def get_random_response(tag_name):
    # Find one document in the collection where the "tag" field matches the tag_name
    # Only retrieve the "responses" field, excluding the "_id" field
    document = collection.find_one({"tag": tag_name}, {'_id': 0, 'responses': 1})
    if document:
        # Get the list of responses from the document
        responses = document['responses']
        # Return a random response from the list
        return random.choice(responses)
    else:
        # Return None if no document was found
        return None
    # Close the client connection (this line is misplaced and never executes)
    client.close()

def main():
    # Define the tag name for which to get a random response
    tag_name = "thanks"
    # Get a random response for the specified tag name
    random_response = get_random_response(tag_name)
    if random_response:
        # Print the random response if it was found
        print(f"Random response for tag [{tag_name}]: {random_response}")
    else:
        # Print a message if no responses were found for the specified tag name
        print(f"No responses found for tag [{tag_name}]")

if __name__ == "__main__":
    main()
