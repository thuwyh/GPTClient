from typing import List
import os
from gpt_client import GPTClient, TGITask
from gpt_client.result_parser import parse_results_as_strings

client = GPTClient(
    tgi_addr="http://184.105.4.137:8300"
)

prompt = """[INST] We are building a RAG QA system in Newsbreak APP. Help me to process a collected real user query(conversation history is also provided).
First determine whether the query is about a specific location or not(if so, we will add location constrains in the following retrieval process). Then rewrite the query to potentially improve the search result if needed. Rewrite only if:
1. conversation history is not empty. Then replace all references in the query with specific information from the history to make the query self-contained.
2. it is likely to be a local query but without a specific location mentioned. Then you may add the user location to the query.
When rewriting, try not to alter the original query too much, and the rewritten query should not be too long.
Then classify the query into at most two(should be only one for most cases) of the provided intents. You should also take the previous conversions into account if provided. Different intents will use different search/API to answer.
Finally, extract slots.
Conversation History:

User Query: Is Trump's New York fraud truly "victimless"?
User Location: Chillicothe, Ohio

Please provide your result as the JSON format below:
{"is_local": true or false, "rewritten query": "self-contained query", "intents": ["intent1", "intent2"], "slots": {"key": "value"}} [/INST]"""

tasks: List[TGITask] = []
for i in range(30):
    tasks.append(TGITask(
        prompt=prompt
    ))

results = client.run_chat_completion_tasks(tasks)
for r in results:
    print(r)