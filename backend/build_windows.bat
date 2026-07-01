cd /d "%~dp0"

rem Build inside a clean, isolated virtual environment so PyInstaller's
rem --collect-all flags only see this project's actual dependencies, not
rem whatever else happens to be installed globally on this machine.
rmdir /s /q build-venv
python -m venv build-venv
call build-venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -r requirements.txt

rmdir /s /q build
rmdir /s /q dist
del /q *.spec

pyinstaller ^
  --onefile ^
  --windowed ^
  --name "OLSEDG-Helper" ^
  --hidden-import=tkinter ^
  --hidden-import=datasets ^
  --hidden-import=pandas ^
  --hidden-import=docx ^
  --hidden-import=requests ^
  --collect-all datasets ^
  --collect-all pandas ^
  --collect-all pyarrow ^
  --exclude-module torch ^
  --exclude-module torchvision ^
  --exclude-module torchaudio ^
  --exclude-module tensorflow ^
  --exclude-module jax ^
  --exclude-module PyQt5 ^
  --exclude-module PySide2 ^
  --exclude-module PySide6 ^
  --exclude-module botocore ^
  --exclude-module boto3 ^
  --exclude-module s3fs ^
  --exclude-module polars ^
  --exclude-module scipy ^
  --exclude-module sklearn ^
  --exclude-module skimage ^
  --exclude-module selenium ^
  --exclude-module playwright ^
  --exclude-module notebook ^
  --exclude-module jupyterlab ^
  --exclude-module jupyter ^
  --exclude-module ipykernel ^
  --exclude-module matplotlib ^
  --exclude-module astropy ^
  --exclude-module bokeh ^
  --exclude-module panel ^
  --exclude-module dask ^
  --exclude-module distributed ^
  --exclude-module transformers ^
  app.py

call build-venv\Scripts\deactivate.bat

pause
