#!/bin/bash

pip install -r requirements.txt

pyinstaller \
  --windowed \
  --name "OLSEDG Helper" \
  app.py

echo "Done. Mac app created at: dist/OLSEDG Helper.app"
