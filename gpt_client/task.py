from typing import List
from hashlib import md5
from pydantic import BaseModel, PrivateAttr


class Task(BaseModel):

    messages: List[dict]
    temperature: float = 0.7
    # top_k: int = 40
    # top_p: float = 1
    model: str = "gpt-4-turbo-preview"
    # frequency_penalty: float = 0
    # response_format
    task_id: str = PrivateAttr()

    def get_unique_id(self):
        md5_id = md5(
            f"{str(self.messages)}_{self.model}_{self.temperature}".encode()
        ).hexdigest()
        return md5_id

    def model_post_init(self, __context) -> None:
        self.task_id = self.get_unique_id()


if __name__ == "__main__":
    task = Task(messages=[{"hello": "world"}])
    print(task.task_id)
