# SEDG — Synthetic Experimental Data Generator

SEDG helps behavioral researchers generate synthetic participant responses for experiments using local and cloud-based LLMs, so stimuli, survey items, and theoretical predictions can be piloted before costly human-subject data collection.

**Live app:** https://synthstudy.vercel.app/

## Features

- Cloud providers: OpenAI, Gemini, Claude, and OpenRouter, using your own API key
- Local models via Ollama, through the companion "OLSEDG Helper" app
- Personas: HuggingFace Twin-2K-500 digital twins, custom Excel upload, or no persona
- Discrete, continuous, or text response scales
- Export to xlsx and docx
- Optional random seed for reproducible runs
- Provenance columns marking output as LLM-synthetic

## Use the web app

Open https://synthstudy.vercel.app/ — no installation required.

## Run the local helper

Download a prebuilt "OLSEDG Helper" app from [GitHub Releases](https://github.com/siva-insights/synthstudy.github.io/releases), or run it from source:

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app
```

**macOS note:** the prebuilt app is unsigned. If macOS blocks it, remove the quarantine attribute:

```bash
xattr -dr com.apple.quarantine "OLSEDG Helper.app"
```

## Documentation

- [SEDG_User_Guide.docx](SEDG_User_Guide.docx)
- [SEDG_Parameters.docx](SEDG_Parameters.docx)

## Reproducibility & responsible use

Set a random seed to reproduce a run. All generated output is labeled as synthetic. Synthetic responses are meant for piloting and pretesting — they are not a substitute for human subjects in confirmatory research.

## Tests

```bash
cd backend
pytest
```

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE.txt](LICENSE.txt).

## Authors

- Siva Shanmugam Mariappan ([ORCID 0000-0001-8200-3579](https://orcid.org/0000-0001-8200-3579)) — The University of Texas at San Antonio
- Ashwin Malshe ([ORCID 0000-0002-3429-4268](https://orcid.org/0000-0002-3429-4268)) — The University of Texas at San Antonio
