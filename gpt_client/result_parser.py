import json
from typing import List
from openai.types.chat import ChatCompletion

def parse_results_as_strings(results: List[ChatCompletion], default_value=''):
    ret = []
    for r in results:
        try:
            ret.append(r.choices[0].message.content)
        except:
            ret.append(default_value)
    return ret

def parse_results_as_jsons(results: List[ChatCompletion], default_value=None):
    ret = []
    for r in results:
        try:
            ret.append(json.loads(r.choices[0].message.content))
        except:
            ret.append(default_value)
    return ret