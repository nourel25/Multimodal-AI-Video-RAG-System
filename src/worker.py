"""ARQ worker: runs processing jobs queued by the API.

Start with:  arq worker.WorkerSettings
"""

import asyncio

from arq.connections import RedisSettings

from controllers.ProcessController import ProcessController
from core.container import ServiceContainer
from helpers.config import get_settings
from helpers.logging_config import setup_logging
from stores.jobs.JobQueue import PROCESS_JOB_FUNCTION

settings = get_settings()


async def startup(ctx):
    setup_logging(settings.LOG_LEVEL)
    container = await ServiceContainer.create(settings, with_job_queue=False)
    # Load models before the first job so its progress reflects real work.
    await asyncio.to_thread(lambda: (container.transcriber, container.embedder))
    ctx["container"] = container


async def shutdown(ctx):
    await ctx["container"].close()


async def process_job(ctx, job_id: str):
    await ProcessController(ctx["container"]).run_job(job_id)


process_job.__name__ = PROCESS_JOB_FUNCTION


class WorkerSettings:
    functions = [process_job]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)
    max_jobs = 1  # transcription + embedding saturate the CPU/GPU
    job_timeout = 3 * 60 * 60
    max_tries = 1  # failures are recorded per video; retrying the whole job would redo work
