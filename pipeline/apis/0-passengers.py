#!/usr/bin/env python3
"""
    By using the Swapi API, create a method that returns the list of
    ships that can hold a given number of passengers:
"""
import requests


def availableShips(passengerCount):
    """
    """
    url = "https://swapi.dev/api/starships/"
    ships = []

    while url:
        response = requests.get(url)
        data = response.json()
        for ship in data['results']:
            try:
                passengers = ship['passengers'].replace(',', '')
                if passengers.isdigit() and int(passengers) >= passengerCount:
                    ships.append(ship['name'])
            except (ValueError, TypeError):
                continue
        url = data['next']

    return ships


if __name__ == "__main__":
    ships = availableShips(4)
    for ship in ships:
        print(ship)
