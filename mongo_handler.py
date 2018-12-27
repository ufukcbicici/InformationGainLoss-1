import pymongo


class MongoHandler():
    client = pymongo.MongoClient('localhost', 27017)
    db = client["Experiments"]
    collection = db["siu_experiments"]

    def __init__(self):
        self.experiment_id = self.collection.insert_one({}).inserted_id

    def get_experiment_id(self):
        return self.experiment_id

    def save_experiment(self, training_parameters):
        self.collection.update_one({"_id": self.experiment_id}, {"$set": training_parameters})
