from typing import List, Union
from time import perf_counter
from pathlib import Path
import asyncio
import openai
from openai.types.chat import ChatCompletion
from diskcache import Cache
from platformdirs import user_cache_dir
import logging
import coloredlogs
from tqdm.auto import tqdm
from gpt_client.types import Task, TGITask
from gpt_client.utils import get_mean
from huggingface_hub import AsyncInferenceClient
from pydantic import BaseModel
class GPTClient:

    def __init__(
        self,
        api_key=None,
        base_url=None,
        tgi_addr=None,
        timeout=60,
        cache_name: str = None,
        log_level: int = logging.INFO,
    ) -> None:
        if tgi_addr is None:
            self._client = openai.AsyncOpenAI(
                api_key=api_key, base_url=base_url, timeout=timeout
            )
        else:
            self._client = AsyncInferenceClient(
            model=tgi_addr, timeout=timeout
        )
        if cache_name is not None:
            cachedir = user_cache_dir("GPTClient", "thuwyh")
            cache_root = Path(cachedir) / cache_name
            self.cache = Cache(str(cache_root))
        else:
            self.cache = None

        self.logger = logging.getLogger("GPTClient")
        self.logger.setLevel(log_level)
        coloredlogs.install(level=log_level, logger=self.logger)

    async def __run_single_chat_completion_task(
        self, task: Union[Task, TGITask], sem: asyncio.Semaphore, p_bar: tqdm = None, check_fn: Union[callable,None]=None
    ) -> Union[ChatCompletion, str]:
        if self.cache is not None:
            # try to load cached result
            if task._task_id in self.cache:
                if check_fn is not None:
                    valid_cache = check_fn(self.cache[task._task_id])
                else:
                    valid_cache = True
                if valid_cache:
                    task.from_cache = True
                    task.finished = True
                    self.logger.info(f"hit cache for task: {task._task_id}")
                    if p_bar is not None:
                        p_bar.update(1)
                    return self.cache[task._task_id]
                else:
                    self.logger.info(f"cache result is not valid: {task._task_id}")
        start = perf_counter()
        
        async with sem:
            for _ in range(3):
                if isinstance(task, Task):
                    
                    if task.response_format is not None and not isinstance(task.response_format, dict):
                        response = await self._client.beta.chat.completions.parse(
                            messages=task.messages,
                            model=task.model,
                            temperature=task.temperature,
                            response_format=task.response_format,
                        )
                    else:
                        response = await self._client.chat.completions.create(
                            **task.dump_for_openai_client()
                        )
                else:
                    response = await self._client.text_generation(
                    **task.dump_for_tgi_client()
                )
                if check_fn is not None:
                    if check_fn(response):
                        break
                    self.logger.warning("check fail!")
                    print(task.messages)
                else:
                    print(response.choices[0].message.content)
                    break

        # try to update cache
        task.finished = True
        if self.cache is not None:
            self.cache[task._task_id] = response.model_dump()
        task.run_time = perf_counter() - start
        self.logger.debug(f"{task._task_id} finished in {task.run_time:.3f}s.")
        if p_bar is not None:
            p_bar.update(1)
        return response

    def __get_run_status(self, tasks, results: List[Union[ChatCompletion, str]]):
        error_count = sum([1 for task in tasks if not task.finished])
        cache_hit = sum([1 for task in tasks if task.from_cache])
        run_times = [task.run_time for task in tasks if task.run_time is not None]
        mean_run_time = get_mean(run_times)
        self.logger.info(
            f"{len(tasks)-error_count} succeed, {error_count} fail, {cache_hit} hit cache | mean run time {mean_run_time:.3f}s."
        )
        if isinstance(self._client, openai.AsyncOpenAI):
            completion_tokens, prompt_tokens, total_tokens = 0, 0, 0
            for r in results:
                if isinstance(r, ChatCompletion):
                    completion_tokens += r.usage.completion_tokens
                    prompt_tokens += r.usage.prompt_tokens
                    total_tokens += r.usage.total_tokens
            self.logger.info(
                f"prompt tokens: {prompt_tokens}, completion tokens: {completion_tokens}, total tokens: {total_tokens}"
            )

    def run_chat_completion_tasks(
        self, tasks: List[Union[Task, TGITask]], concurrent_num: int = 5, show_progress_bar: bool = True, check_fn: Union[callable,None]=None
    ) -> List[ChatCompletion]:
        start = perf_counter()
        self.logger.info(f"Processing starts. Total tasks: {len(tasks)}")
        
        async def run_with_semaphore():
            a_tasks = []
            sem = asyncio.Semaphore(concurrent_num)
            p_bar = tqdm(total=len(tasks)) if show_progress_bar else None
            for t in tasks:
                a_tasks.append(
                    asyncio.create_task(
                        self.__run_single_chat_completion_task(task=t, sem=sem, p_bar=p_bar, check_fn=check_fn)
                    )
                )
            return await asyncio.gather(*a_tasks, return_exceptions=True)
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()
        results = loop.run_until_complete(run_with_semaphore())

        self.logger.info(f"all tasks finished in {perf_counter()-start:.3f}s")
        self.__get_run_status(tasks, results)
        return results
