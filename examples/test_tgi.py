from typing import List
import os
from gpt_client import GPTClient, TGITask
from gpt_client.result_parser import parse_results_as_strings

client = GPTClient(
    tgi_addr="http://184.105.4.137:8300"
)

tasks: List[TGITask] = []
for i in range(30):
    tasks.append(TGITask(
        prompt="hello"
    ))

results = client.run_chat_completion_tasks(tasks)
for r in results:
    print(r)