#!/usr/bin/env python3
""" By using the (unofficial) SpaceX API, write a script
    that displays the first launch with these information:
"""
import requests
from datetime import datetime


if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4/launches/upcoming'

    response = requests.get(url)
    launches_sorted = sorted(response.json(), key=lambda x: x["date_unix"])

    date_unix = launches_sorted[0]["date_unix"]

    for launch in response.json():
        if launch["date_unix"] == date_unix:
            launch_name = launch["name"]
            date = launch["date_local"]
            rocket_id = launch["rocket"]
            launchpad_id = launch["launchpad"]
            break

    rock_name = requests.get("https://api.spacexdata.com/v4/rockets/"
                             + rocket_id).json()["name"]
    launchpad_name = requests.get("https://api.spacexdata.com/v4/launchpads/"
                                  + launchpad_id).json()["name"]
    launchpad_loc = requests.get("https://api.spacexdata.com/v4/launchpads/"
                                 + launchpad_id).json()["locality"]

    print("{} ({}) {} - {} ({})".format(
        launch_name, date, rock_name, launchpad_name, launchpad_loc))
