# HPCBot

## Install

```bash
pip install -e .
```

## Chat

```bash
hpcbot assist --config configs/config.yaml "Hello"
```

## Generate QA

```bash
hpcbot generate --config configs/config.yaml
```

### Environment Variables
To use OpenAI, you need to provide your own API key by setting the `OPENAI_API_KEY` environment variable.
Setting `ANONYMIZED_TELEMETRY=False` will disable data collection by Chroma. Check out https://docs.trychroma.com/docs/overview/telemetry for more details.

