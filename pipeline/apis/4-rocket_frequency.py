#!/usr/bin/env python3
"""
    By using the (unofficial) SpaceX API, write a script
    that displays the number of launches per rocket
"""
import requests
import json

if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v3/launches'

    response = requests.get(url)
    launches = json.loads(response.content)

    rockets = {}

    for launch in launches:
        rocket_name = launch['rocket']['rocket_name']
        if rocket_name not in rockets:
            rockets[rocket_name] = 1
        else:
            rockets[rocket_name] += 1

    sorted_rockets = sorted(rockets.items(), key=lambda x: x[1], reverse=True)
    for rocket, count in sorted_rockets:
        print(f"{rocket}: {count}")
