#!/usr/bin/env python3
""" Write a Python function that returns all students
    sorted by average score
"""


def top_students(mongo_collection):
    """
    mongo_collection will be the pymongo collection object
    The top must be ordered
    The average score must be part of each item returns with key = averageScore
    """
    pipeline = [
        {
            "$project": {
                "name": 1,
                "topics": 1,
                "averageScore": { "$avg": "$topics.score" }
            }
        },
        {
            "$sort": { "averageScore": -1 }
        }
    ]
    
    return list(mongo_collection.aggregate(pipeline))
