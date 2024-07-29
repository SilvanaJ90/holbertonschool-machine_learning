#!/usr/bin/env python3
import requests
import sys
from datetime import datetime

def get_user_location(url):
    response = requests.get(url)

    if response.status_code == 404:
        print("Not found")
    elif response.status_code == 403:
        reset_time = int(response.headers.get('X-Ratelimit-Reset', 0))
        reset_datetime = datetime.fromtimestamp(reset_time)
        minutes_left = int((reset_datetime - datetime.now()).total_seconds() / 60)
        print(f"Reset in {minutes_left} min")
    elif response.status_code == 200:
        user_data = response.json()
        print(user_data.get('location', 'Location not available'))
    else:
        print("Unexpected error")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <GitHub API URL>")
        sys.exit(1)

    url = sys.argv[1]
    get_user_location(url)
