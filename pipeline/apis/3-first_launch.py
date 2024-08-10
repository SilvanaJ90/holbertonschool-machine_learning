#!/usr/bin/env python3
"""
    By using the (unofficial) SpaceX API, write a script
    that displays the first launch with these information:
"""
import requests

if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4/launches/upcoming'

    response = requests.get(url)
    response.raise_for_status()
    launches = response.json()

    launches_sorted = sorted(launches, key=lambda x: x["date_unix"])

    first_launch = launches_sorted[0]

    launch_name = first_launch["name"]
    date = first_launch["date_local"]
    rocket_id = first_launch["rocket"]
    launchpad_id = first_launch["launchpad"]

    rocket_response = requests.get(
        f"https://api.spacexdata.com/v4/rockets/{rocket_id}")
    rocket_response.raise_for_status()
    rocket_name = rocket_response.json()["name"]

    launchpad_response = requests.get(
        f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}")
    launchpad_response.raise_for_status()
    launchpad = launchpad_response.json()
    launchpad_name = launchpad["name"]
    launchpad_loc = launchpad["locality"]

    print("{} ({}) {} - {} ({})".format(launch_name, date, rocket_name, launchpad_name, launchpad_loc))
