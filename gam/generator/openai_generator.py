import time
import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

from gam.generator.base import AbsGenerator
from gam.config import OpenAIGeneratorConfig


class OpenAIGenerator(AbsGenerator):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.model_name = config.get("model_name", "gpt-4o-mini")
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")

        self.n = config.get("n", 1)
        self.temperature = config.get("temperature", 0.0)
        self.top_p = config.get("top_p", 1.0)
        self.max_tokens = config.get("max_tokens", 300)

        self.thread_count = config.get("thread_count")
        self.system_prompt = config.get("system_prompt")

        self.timeout = config.get("timeout", 60.0)
        self.use_schema = config.get("use_schema", False)

        if self.api_key:
            os.environ["OPENAI_API_KEY"] = self.api_key

        if self.base_url:
            os.environ["OPENAI_BASE_URL"] = self.base_url

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url.rstrip("/") if self.base_url else None
        )

        self._cclient = (
            self._client.with_options(timeout=self.timeout)
            if hasattr(self._client, "with_options")
            else self._client
        )

    def _build_messages(
        self,
        prompt: Optional[str],
        messages: Optional[List[Dict[str, str]]],
    ):

        if prompt is None and not messages:
            raise ValueError("Either prompt or messages is required.")

        if prompt is not None and messages:
            raise ValueError("Pass either prompt or messages, not both.")

        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        if self.system_prompt and not any(m["role"] == "system" for m in messages):
            messages = [{"role": "system", "content": self.system_prompt}] + messages

        return messages

    def _extract_json(self, text: str):

        try:
            start = text.find("{")
            end = text.rfind("}")

            if start == -1 or end == -1:
                return None

            return json.loads(text[start:end + 1])

        except Exception:
            return None

    def generate_single(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        msgs = self._build_messages(prompt, messages)

        response_format = None

        if schema and self.use_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "auto_schema",
                    "schema": schema,
                    "strict": True,
                },
            }

        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": msgs,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "n": self.n,
        }

        if response_format:
            params["response_format"] = response_format

        if extra_params:
            params.update(extra_params)

        retry = 0

        while True:
            try:
                resp = self._cclient.chat.completions.create(**params)
                break
            except Exception as e:
                retry += 1
                print(e, "retry:", retry)

                if retry > 3:
                    raise e

                time.sleep(5)

        text = ""

        try:
            text = resp.choices[0].message.content or ""
        except Exception:
            pass

        text = text.split("</think>")[-1]

        out = {
            "text": text,
            "json": None,
            "response": resp.model_dump(),
        }

        if schema:
            out["json"] = self._extract_json(text)

        return out

    def generate_batch(
        self,
        prompts: Optional[List[str]] = None,
        messages_list: Optional[List[List[Dict[str, str]]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:

        if prompts is None and not messages_list:
            raise ValueError("Either prompts or messages_list is required.")

        if prompts is not None and messages_list:
            raise ValueError("Pass either prompts or messages_list, not both.")

        if prompts is not None:

            if isinstance(prompts, str):
                prompts = [prompts]

            messages_list = [
                [{"role": "user", "content": p}] for p in prompts
            ]

        thread_count = self.thread_count or cpu_count()

        def worker(msgs):
            return self.generate_single(
                messages=msgs,
                schema=schema,
                extra_params=extra_params,
            )

        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            results = list(
                tqdm(
                    executor.map(worker, messages_list),
                    total=len(messages_list),
                )
            )

        return results

    @classmethod
    def from_config(cls, config: OpenAIGeneratorConfig):
        return cls(config.__dict__)