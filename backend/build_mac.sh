#!/bin/bash

pip install -r requirements.txt

rm -rf build
rm -rf dist
rm -f *.spec

pyinstaller \
  --windowed \
  --name "OLSEDG Helper" \
  --hidden-import=datasets \
  --hidden-import=pandas \
  --hidden-import=docx \
  --hidden-import=requests \
  --collect-all datasets \
  --collect-all pandas \
  --collect-all pyarrow \
  app.py

echo "Done. Mac app created at: dist/OLSEDG Helper.app"
