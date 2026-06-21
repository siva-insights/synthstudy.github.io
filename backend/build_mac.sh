#!/bin/bash

pip install -r requirements.txt

pyinstaller --onefile --name OLSEDG-Helper app.py

echo "Done. Helper created at: dist/OLSEDG-Helper"
