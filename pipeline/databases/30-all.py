#!/usr/bin/env python3
""" Python function that lists all documents in a collection """


def list_all(mongo_collection):
    """
    Return an empty list if no document in the collection
    mongo_collection will be the pymongo collection object
    """
    if mongo_collection is None:
        return []
    return list(mongo_collection.find())
