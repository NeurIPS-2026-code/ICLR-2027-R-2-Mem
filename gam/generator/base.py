from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class AbsGenerator(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def generate_single(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        生成单个响应

        返回格式:
        {
            "text": str,
            "json": dict | None,
            "response": dict
        }

        注意：
        temperature, max_tokens 等参数已在配置中设置
        """
        pass

    @abstractmethod
    def generate_batch(
        self,
        prompts: Optional[List[str]] = None,
        messages_list: Optional[List[List[Dict[str, str]]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量生成响应

        返回格式:
        [
            {
                "text": str,
                "json": dict | None,
                "response": dict
            }
        ]
        """
        pass