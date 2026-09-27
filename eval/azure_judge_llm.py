"""DeepEvalBaseLLM wrapper around this project's existing, already-verified-live
Azure client (src/azure_foundry.py) -- reused as-is, no LangChain, no new client
code. This is the JUDGE model, deliberately separate from the production
generation model (Groq openai/gpt-oss-20b, see src/generate.py) to avoid a model
grading its own answers.
"""
from deepeval.models.base_model import DeepEvalBaseLLM

from src.azure_foundry import get_azure_client, AZURE_OPENAI_DEPLOYMENT


class AzureJudgeLLM(DeepEvalBaseLLM):
    def __init__(self):
        super().__init__(model=AZURE_OPENAI_DEPLOYMENT)

    def load_model(self):
        return get_azure_client()

    def get_model_name(self) -> str:
        return f"azure:{AZURE_OPENAI_DEPLOYMENT}"

    def generate(self, prompt: str) -> str:
        raise NotImplementedError("Sync path unused -- eval harness is async throughout.")

    async def a_generate(self, prompt: str) -> str:
        client = self.load_model()
        response = await client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content
