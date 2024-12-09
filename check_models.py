#!/usr/bin/env python3
import requests
import json

def main():
    try:
        response = requests.get('http://localhost:8000/v1/models')
        print(json.dumps(response.json(), indent=2))
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")

if __name__ == "__main__":
    main()
