# Python bindings

Veld ships Python bindings via PyO3 + maturin. Install:

```sh
pip install veld
```

## Basic usage

```python
import veld

# Spin up an embedded veld instance (process-local; no separate daemon)
client = veld.Client(data_dir="~/.veld-python")

# Remember
mem_id = client.remember(
    content="The auth middleware uses API-key headers",
    importance=0.8,
    tags=["auth", "architecture"],
)

# Recall
results = client.recall(query="how does auth work")
for r in results:
    print(r.content, r.score)

# Proactive context (for use at the start of an LLM turn)
context = client.proactive_context()
```

## Build from source

If `pip install` doesn't have a wheel for your platform:

```sh
git clone https://github.com/Portll/veld
cd veld
maturin build --release --features python
pip install target/wheels/veld-*.whl
```

## ONNX embedder

The Python bindings ship with an embedded ONNX runtime for MiniLM-L6-v2. No
external embedding server required for the default config.

## Async API

```python
import asyncio
import veld

async def main():
    client = veld.AsyncClient(data_dir="~/.veld-python")
    results = await client.recall(query="...")
    return results

asyncio.run(main())
```

## See also

- [Configuration reference](../reference/config.md)
- [Architecture overview](../architecture/overview.md)
