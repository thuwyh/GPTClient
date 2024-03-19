from typing import List
from time import perf_counter
from pathlib import Path
import asyncio
import openai
from openai.types.chat import ChatCompletion
from diskcache import Cache
from platformdirs import user_cache_dir
import logging
import coloredlogs
from gpt_client.task import Task


class GPTClient:

    def __init__(
        self,
        api_key,
        base_url=None,
        timeout=60,
        cache_name: str = None,
        log_level: int = logging.INFO,
    ) -> None:
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        if cache_name is not None:
            cachedir = user_cache_dir("GPTClient", "thuwyh")
            cache_root = Path(cachedir) / cache_name
            self.cache = Cache(str(cache_root))
        else:
            self.cache = None

        self.logger = logging.getLogger("real_search")
        self.logger.setLevel(log_level)
        coloredlogs.install(level=log_level, logger=self.logger)

        self.cache_hit = 0

    async def __run_single_task(
        self, task: Task, sem: asyncio.Semaphore
    ) -> ChatCompletion:
        if self.cache is not None:
            # try to load cached result
            if task.task_id in self.cache:
                self.cache_hit += 1
                self.logger.info(f"hit cache for task: {task.task_id}")
                return self.cache[task.task_id]
        start = perf_counter()
        async with sem:
            response = await self._client.chat.completions.create(
                messages=task.messages, model=task.model, temperature=task.temperature
            )

        # try to update cache
        if self.cache is not None:
            self.cache[task.task_id] = response
        self.logger.debug(f"{task.task_id} finished in {perf_counter()-start:.3f}s.")
        return response

    async def __run_with_semaphore(
        self, tasks: List[Task], concurrent_num: int
    ) -> List[ChatCompletion]:
        a_tasks = []
        sem = asyncio.Semaphore(concurrent_num)
        for t in tasks:
            a_tasks.append(asyncio.create_task(self.__run_single_task(task=t, sem=sem)))

        results = await asyncio.gather(*a_tasks, return_exceptions=True)
        return results

    def run_tasks(
        self, tasks: List[Task], concurrent_num: int = 5
    ) -> List[ChatCompletion]:
        start = perf_counter()
        self.cache_hit = 0
        self.logger.info(f"Processing starts. Total tasks: {len(tasks)}")
        results = asyncio.run(
            self.__run_with_semaphore(tasks, concurrent_num=concurrent_num)
        )
        error_count = sum([1 for r in results if not isinstance(r, ChatCompletion)])
        self.logger.info(
            f"all tasks finished in {perf_counter()-start:.3f}s. {len(tasks)-error_count} succeed, {error_count} fail, {self.cache_hit} hit cache."
        )
        return results
