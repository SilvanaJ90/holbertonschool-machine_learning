#!/usr/bin/env python3
"""
    By using the (unofficial) SpaceX API, write a script
    that displays the number of launches per rocket
"""
import requests
import json


if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4/launches'

    response = requests.get(url)
    launches = json.loads(response.content)

    rockets = {}

    for launch in launches:
        rocket_id = launch['rocket']
        rocket_url = f'https://api.spacexdata.com/v4/rockets/{rocket_id}'
        rocket_response = requests.get(rocket_url)
        rocket_response.raise_for_status()
        rocket_name = rocket_response.json()['name']

        if rocket_name not in rockets:
            rockets[rocket_name] = 1
        else:
            rockets[rocket_name] += 1

    sorted_rockets = sorted(rockets.items(), key=lambda x: x[1], reverse=True)
    for rocket, count in sorted_rockets:
        print("{}: {}".format(rocket, count))
