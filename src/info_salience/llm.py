from pathlib import Path

import litellm
import vllm
from vllm.sampling_params import GuidedDecodingParams
from pydantic import BaseModel


class VLLMGenerator:
    def __init__(self, model, **kwargs):
        self.model = model
        self.engine = 'vllm'
        self.llm = vllm.LLM(model, **kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, messages, schema: BaseModel = None, **kwargs):
        if schema:
            guided_decoding = GuidedDecodingParams(json=schema.model_json_schema())
        else:
            guided_decoding = None

        params = vllm.SamplingParams(guided_decoding=guided_decoding, **kwargs)
        prompts = self.tokenizer.apply_chat_template(messages, tokenize=False)
        responses = self.llm.generate(prompts, params)
        # results shape: (prompts, n) where n is number of generations
        results = [
            [output.text for output in response.outputs] for response in responses
        ]
        return results


class LitellmGenerator:
    def __init__(self, model, caching=False, report_costs=False, disk_cache_dir=None):
        self.model = model
        self.engine = 'litellm'
        self.caching = caching
        if caching:
            from litellm.caching import Cache

            if disk_cache_dir:
                Path(disk_cache_dir).mkdir(exist_ok=True, parents=True)
            litellm.cache = Cache("disk", disk_cache_dir=disk_cache_dir)
        self.report_costs = report_costs

    def generate(self, messages, **kwargs):
        responses = litellm.batch_completion(
            model=self.model,
            messages=messages,
            caching=self.caching,
            **kwargs,
        )
        if self.report_costs:
            total = 0
            for response in responses:
                total += litellm.completion_cost(response)
            print(f"Batch cost: ${total:.4f}")

        results = [
            [choice["message"]["content"] for choice in response["choices"]]
            for response in responses
        ]
        return results
