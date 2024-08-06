#!/usr/bin/env python3
""" By using the (unofficial) SpaceX API, write a script
    that displays the first launch with these information:
"""
import requests
from datetime import datetime


def fetch_data(url):
    """Fetch data from the given URL."""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4/launches/upcoming'

    launches = fetch_data(url)
    launches_sorted = sorted(launches, key=lambda x: x["date_unix"])

    # Get the first launch
    first_launch = launches_sorted[0]

    launch_name = first_launch["name"]
    date = first_launch["date_local"]
    rocket_id = first_launch["rocket"]
    launchpad_id = first_launch["launchpad"]

    # Fetch rocket and launchpad details
    rocket_name = fetch_data(
        f"https://api.spacexdata.com/v4/rockets/{rocket_id}")["name"]
    launchpad = fetch_data(
        f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}")
    launchpad_name = launchpad["name"]
    launchpad_loc = launchpad["locality"]

    print("{} ({}) {} - {} ({})".format(
        launch_name, date, rocket_name, launchpad_name, launchpad_loc))
