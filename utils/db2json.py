import pymongo
import json
from pymongo import MongoClient

# Kết nối đến MongoDB
connect_str = "mongodb://Nhan:1@127.0.0.1:27017/tensorbot_nlp"
client = MongoClient(connect_str)
db = client.tensorbot_nlp
collection = db["dialog_nlp"]

# Đọc dữ liệu từ MongoDB
data = list(collection.find({}, {'_id': 0}))

# Chuyển dữ liệu thành định dạng JSON
json_data = json.dumps(data, indent=4)

# In ra dữ liệu JSON
print(json_data["tag"])
