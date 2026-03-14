import pytest

from classification.ollama import OllamaClient, OllamaError


@pytest.mark.integration
def test_ollama_smoke_if_available() -> None:
    client = OllamaClient(timeout_seconds=5)
    try:
        models = client.list_models()
    except OllamaError:
        pytest.skip("Ollama is not running locally")
    if "qwen3.5:0.8b" not in models:
        pytest.skip("qwen3.5:0.8b is not installed in local Ollama")

    generation = client.generate_json(
        model="qwen3.5:0.8b",
        system_prompt="Return JSON only.",
        prompt='Return {"label":"cash_withdrawal"} exactly.',
        options={"temperature": 0, "num_predict": 16},
        schema={
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
            "additionalProperties": False,
        },
    )

    assert generation.raw_output
