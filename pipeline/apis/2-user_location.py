#!/usr/bin/env python3
""" By using the GitHub API, write a script that
    prints the location of a specific user:
"""
import requests
import sys
from datetime import datetime


if __name__ == '__main__':
    """ Doc """
    url = sys.argv[1]
    response = requests.get(url)
    print(response)
