import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

PROCESS_JOB_FUNCTION = "process_job"


class JobQueueInterface(ABC):
    @abstractmethod
    async def enqueue_process(self, job_id: str) -> None: ...

    async def close(self) -> None:
        return None


class ArqJobQueue(JobQueueInterface):
    """Hands jobs to the ARQ worker (see worker.py) through Redis."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.pool = None
        self._connect_lock = asyncio.Lock()

    async def connect(self):
        from arq import create_pool
        from arq.connections import RedisSettings

        self.pool = await create_pool(RedisSettings.from_dsn(self.redis_url))

    async def enqueue_process(self, job_id: str) -> None:
        async with self._connect_lock:  # concurrent first calls must not create two pools
            if self.pool is None:
                await self.connect()
        # _job_id makes enqueueing idempotent: the same job can't run twice concurrently.
        await self.pool.enqueue_job(PROCESS_JOB_FUNCTION, job_id, _job_id=job_id)

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.aclose()
            self.pool = None


class InlineJobQueue(JobQueueInterface):
    """Runs jobs as background tasks inside the API process — no Redis needed.

    Fine for local development; jobs are lost if the process restarts.
    """

    def __init__(self, runner: Callable[[str], Awaitable[None]]):
        self.runner = runner
        self._tasks: set[asyncio.Task] = set()
        # One job at a time: transcription and embedding are CPU/GPU bound.
        self._lock = asyncio.Lock()

    async def _run(self, job_id: str) -> None:
        async with self._lock:
            try:
                await self.runner(job_id)
            except Exception:
                logger.exception("Inline job %s crashed", job_id)

    async def enqueue_process(self, job_id: str) -> None:
        task = asyncio.create_task(self._run(job_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def wait_idle(self) -> None:
        """Wait for all queued jobs (used by tests and graceful shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*list(self._tasks), return_exceptions=True)
