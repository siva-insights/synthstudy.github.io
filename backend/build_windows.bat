pip install -r requirements.txt

rmdir /s /q build
rmdir /s /q dist
del /q *.spec

pyinstaller ^
  --onefile ^
  --console ^
  --name "OLSEDG-Helper" ^
  --hidden-import=datasets ^
  --hidden-import=pandas ^
  --hidden-import=docx ^
  --hidden-import=requests ^
  --collect-all datasets ^
  --collect-all pandas ^
  --collect-all pyarrow ^
  app.py

pause
