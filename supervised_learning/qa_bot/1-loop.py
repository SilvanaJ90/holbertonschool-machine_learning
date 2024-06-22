#!/usr/bin/env python3
"""
  that takes in input from the user with the prompt Q:
    and prints A: as a response.
  If the user inputs exit, quit, goodbye, or bye,
    case insensitive, print A: Goodbye and exit.
"""

quit = ['exit', 'quit', 'goodbye', 'bye']

while True:
    question = input('Q: ').lower()
    if question in quit:
        print('A: Goodbye')
        break
    print('A:')
