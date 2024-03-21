from typing import List, Union
from hashlib import md5
from pydantic import BaseModel, PrivateAttr
from openai.types.chat import ChatCompletion


class Task(BaseModel):

    messages: List[dict]
    temperature: float = 0.7
    # top_k: int = 40
    # top_p: float = 1
    model: str = "gpt-4-turbo-preview"
    frequency_penalty: float = 0
    response_format: dict = None
    max_tokens: int = None

    _task_id: str = PrivateAttr()

    finished: bool = False
    from_cache: bool = False
    run_time: float = None

    def get_unique_id(self):
        md5_id = md5(
            f"{str(self.messages)}_{self.model}_{self.temperature}".encode()
        ).hexdigest()
        return md5_id

    def model_post_init(self, __context) -> None:
        self._task_id = self.get_unique_id()

    def dump_for_openai_client(self):
        include=[
                "messages",
                "model",
                "temperature",
                "max_tokens",
                "frequency_penalty"
            ]
        if self.response_format is not None:
            include.append("response_format")
        return self.model_dump(
            include=include
        )


if __name__ == "__main__":
    task = Task(messages=[{"hello": "world"}])
    print(task._task_id)
    print(task.dump_for_openai_client())
