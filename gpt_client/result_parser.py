import json
from typing import List, Union
from openai.types.chat import ChatCompletion

def parse_results_as_strings(results: List[Union[ChatCompletion, str]], default_value=''):
    ret = []
    for r in results:
        try:
            if isinstance(r, ChatCompletion):
                ret.append(r.choices[0].message.content)
            else:
                ret.append(r)
        except:
            ret.append(default_value)
    return ret

def parse_results_as_jsons(results: List[Union[ChatCompletion, str]], default_value=None):
    ret = []
    for r in results:
        try:
            if isinstance(r, ChatCompletion):
                ret.append(json.loads(r.choices[0].message.content))
            else:
                ret.append(json.loads(r))
        except:
            ret.append(default_value)
    return ret