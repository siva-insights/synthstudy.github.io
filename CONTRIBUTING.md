# Contributing

## Bugs and feature requests

Please report bugs and request features via [GitHub Issues](https://github.com/siva-insights/synthstudy.github.io/issues).

## Development setup

The frontend is a single self-contained `index.html` — open it directly in a browser, or serve it statically.

The backend (OLSEDG Helper) uses a virtual environment:

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

## Before opening a PR

Run the backend test suite:

```bash
cd backend
pytest
```

## Workflow

Branch → PR → CI (tests) must pass before merge.
