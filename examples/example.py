from typing import List
import os
from gpt_client import GPTClient, Task
from dotenv import load_dotenv

load_dotenv()

client = GPTClient(
    api_key=os.environ['OPENAI_KEY'],
    base_url=os.environ['OPENAI_BASE_URL'],
    cache_name='example1'
)

tasks: List[Task] = []
for i in range(30):
    tasks.append(Task(
        messages=[{'role':'user', 'content':f'{i}+{i+1}='}]
    ))

results = client.run_chat_completion_tasks(tasks)
for t, r in zip(tasks, results):
    try:
        print(t.messages[0]['content'], r.choices[0].message.content)
    except:
        pass

# outputs:
# 0+1= 0 + 1 = 1
# 1+2= 1 + 2 = 3
# 2+3= 2 + 3 = 5
# 3+4= 3 + 4 = 7
# 4+5= 4 + 5 = 9
# 5+6= 5 + 6 = 11
# 6+7= 6 + 7 = 13
# 7+8= 7 + 8 = 15
# 8+9= 8 + 9 = 17
# 9+10= 9 + 10 = 19
# 10+11= 10 + 11 = 21
# 11+12= 11 + 12 = 23
# 12+13= 12 + 13 = 25
# 13+14= 13 + 14 = 27
# 14+15= 14 + 15 = 29
# 15+16= 15 + 16 = 31
# 16+17= 16 + 17 = 33
# 17+18= 17 + 18 = 35
# 18+19= 18 + 19 = 37
# 19+20= 19 + 20 = 39
# 20+21= 20 + 21 = 41
# 21+22= 21 + 22 = 43
# 22+23= 22 + 23 = 45
# 23+24= 23 + 24 = 47
# 24+25= 24 + 25 = 49
# 25+26= 25 + 26 = 51
# 26+27= 26 + 27 = 53
# 27+28= 27 + 28 = 55
# 28+29= 28 + 29 = 57
# 29+30= 29 + 30 = 59