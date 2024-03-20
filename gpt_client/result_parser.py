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
