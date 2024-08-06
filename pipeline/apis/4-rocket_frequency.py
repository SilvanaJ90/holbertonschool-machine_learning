#!/usr/bin/env python3
"""

"""
import requests


if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4/launches'
     
    response = requests.get(url)
    launches_sorted = sorted(response.json(), key=lambda x:x["date_unix"])

    date_unix = launches_sorted[-1]["date_unix"]

    for launch in response.json():
        if launch["date_unix"] == date_unix:
            launch_name = launch["name"]
            rocket_id = lauch["date_local"]
            launchpad_id = launch["lauchpad"]
            break
    
    rock_name = requests.get("https://api.spacexdata.com/v4/rockets/" + rocket_id).json()["name"]
    launchpad_name = requests.get("https://api.spacexdata.com/v4/launchpads/" + launchpad_id).json()["name"]
    launchpad_locality = requests.get("https://api.spacexdata.com/v4/launchpads/" + launchpad_id).json()["locality"]
    print(response)