#!/bin/bash

pip install -r requirements.txt

pyinstaller \
  --onefile \
  --name "OLSEDG-Helper" \
  app.py
