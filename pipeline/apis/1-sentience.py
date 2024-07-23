#!/usr/bin/env python3
"""
    By using the Swapi API, create a method that returns the list
    of names of the home planets of all sentient species.
"""
import requests


def sentientPlanets():
    """
    sentient type is either in the classification
    or designation attributes.
    """
    url = "https://swapi.dev/api/species/"
    planets = set()

    while url:
        response = requests.get(url)
        data = response.json()
        for species in data['results']:
            classification = species.get('classification', '').lower()
            designation = species.get('designation', '').lower()
            if 'sentient' in classification or 'sentient' in designation:
                homeworld_url = species.get('homeworld')
                if homeworld_url:
                    planet_response = requests.get(homeworld_url)
                    planet_data = planet_response.json()
                    planets.add(planet_data['name'])

        url = data['next']

    return sorted(planets)
