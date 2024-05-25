from langchain.llms.base import LLM
from typing import Optional, List, Any, Mapping
from langchain.callbacks.manager import CallbackManagerForLLMRun
from http import HTTPStatus
import os
from dashscope import Generation
from dotenv import load_dotenv, find_dotenv

"""
langchain 框架封装了部分大模型，此处使用自定义的通义千问模型
"""
_ = load_dotenv(find_dotenv())


class DashLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        return self.invoke(prompt)

    model: str = "qwen-turbo"
    temperature: float = 0.7

    @property
    def _llm_type(self) -> str:
        return "dashllm"

    def invoke(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        # 如果用户在页面没有输入 api_key，就使用 .env 文件中定义的 key
        user_key = kwargs.get("api_key", None)
        if user_key is None:
            api_key = os.environ['DASHSCOPE_API_KEY']
        else:
            api_key = user_key
        response = Generation.call(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            api_key=api_key
        )
        if response.status_code != HTTPStatus.OK:
            return f"请求失败，失败信息为:{response.message}"
        return response.output.text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model}


if __name__ == '__main__':
    qw = DashLLM()
    print(qw.predict("北京有什么好吃的？"))
