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

    try:
        response = requests.get(url)
        if response.status_code == 200:
            location = response.json().get("location")
            if location:
                print(location)
            else:
                print("Location not available")
        elif response.status_code == 403:
            X_Ratelimit_Reset = response.headers.get("X-Ratelimit-Reset")
            now_time = datetime.now().timestamp()
            diff = (int(X_Ratelimit_Reset) - now_time) / 60
            print("Reset in {} min".format(int(diff)))
        else:
            print("Not found")
    except requests.RequestException as e:
        print(f"Error: {e}")
