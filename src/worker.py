"""ARQ worker: runs processing jobs queued by the API.

Start with:  arq worker.WorkerSettings
"""

import asyncio
import logging

from arq.connections import RedisSettings

from controllers.JobController import JobController
from controllers.ProcessController import ProcessController
from core.container import ServiceContainer
from helpers.config import get_settings
from helpers.logging_config import setup_logging
from models.enums import JobStatus
from stores.jobs.JobQueue import PROCESS_JOB_FUNCTION
from stores.llm import ProviderConfigError

logger = logging.getLogger(__name__)

settings = get_settings()


async def startup(ctx):
    setup_logging(settings.LOG_LEVEL)
    container = await ServiceContainer.create(settings, with_job_queue=False)
    # A job "running" at startup belonged to a worker that died (max_jobs=1, max_tries=1).
    await JobController(container).recover_interrupted([JobStatus.RUNNING])
    try:
        # Load models before the first job so its progress reflects real work.
        await asyncio.to_thread(lambda: (container.transcriber, container.embedder))
    except ProviderConfigError as exc:
        # Keep running: each job will fail its videos with this message instead of the
        # worker crash-looping while the API keeps accepting work.
        logger.error("Provider misconfigured, jobs will fail until fixed: %s", exc)
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
