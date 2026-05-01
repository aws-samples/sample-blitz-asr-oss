"""
Generic batch processing service for micro-batching inference requests.

Provides a reusable batch worker that:
- Monitors a queue for engine-specific requests
- Gathers batches within a timeout window
- Delegates to engine-specific processors
- Distributes results to waiting futures
"""

import asyncio
import logging
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class BatchWorker:
    """Generic batch worker for micro-batching inference requests."""

    def __init__(
        self,
        engine_name: str,
        transcription_queue: asyncio.Queue,
        pending_results: Dict[str, asyncio.Future],
        process_batch_func: Callable[[List[Dict]], List[Dict]],
        cleanup_func: Optional[Callable[[List[Dict], List[Dict]], None]] = None,
        max_batch_size: int = 4,
        batch_timeout_ms: float = 15.0,
    ):
        """
        Initialize batch worker.

        Args:
            engine_name: Name of the engine (e.g., "parakeet", "whisper")
            transcription_queue: Shared queue for all transcription requests
            pending_results: Dictionary mapping job_id to Future objects
            process_batch_func: Function to process a batch of jobs
                - Input: List of job dictionaries
                - Output: List of result dictionaries
            cleanup_func: Optional cleanup function called after each batch
            max_batch_size: Maximum number of requests to batch together
            batch_timeout_ms: Wait up to this many milliseconds to gather a batch
        """
        self.engine_name = engine_name
        self.transcription_queue = transcription_queue
        self.pending_results = pending_results
        self.process_batch_func = process_batch_func
        self.cleanup_func = cleanup_func
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms

    async def run(self) -> None:
        """
        Run the batch worker loop.

        Forever drains queue → batches → processes → returns results.
        """
        print(
            f"🔄 {self.engine_name.capitalize()} batch worker started "
            f"(batch_size={self.max_batch_size}, timeout={self.batch_timeout_ms}ms)"
        )

        while True:
            try:
                # Wait for first request
                job = await self.transcription_queue.get()

                # Filter by engine
                if job["engine"] != self.engine_name:
                    # Put back if not for this worker
                    await self.transcription_queue.put(job)
                    await asyncio.sleep(0.001)  # Brief sleep to avoid busy loop
                    continue

                batch = [job]

                # Gather additional requests within timeout
                deadline = asyncio.get_event_loop().time() + (
                    self.batch_timeout_ms / 1000
                )

                while len(batch) < self.max_batch_size:
                    timeout = deadline - asyncio.get_event_loop().time()
                    if timeout <= 0:
                        break

                    try:
                        next_job = await asyncio.wait_for(
                            self.transcription_queue.get(), timeout
                        )

                        if next_job["engine"] == self.engine_name:
                            batch.append(next_job)
                        else:
                            # Put back if not for this worker
                            await self.transcription_queue.put(next_job)

                    except asyncio.TimeoutError:
                        break

                # Process batch using engine-specific function
                results = self.process_batch_func(batch)

                # Return results to waiting requests
                for job, result in zip(batch, results):
                    future = self.pending_results.get(job["job_id"])
                    if future and not future.done():
                        future.set_result(result)

                    self.transcription_queue.task_done()

                # Run cleanup if provided
                if self.cleanup_func:
                    self.cleanup_func(batch, results)

            except asyncio.CancelledError:
                print(f"🛑 {self.engine_name.capitalize()} batch worker cancelled")
                break

            except Exception as e:
                logger.exception(
                    "%s batch worker error: %s", self.engine_name.capitalize(), e
                )

                # Set exception for all jobs in batch
                for job in batch:
                    future = self.pending_results.get(job["job_id"])
                    if future and not future.done():
                        future.set_exception(Exception(f"Batch processing failed: {e}"))
                    self.transcription_queue.task_done()


def create_batch_worker(
    engine_name: str,
    transcription_queue: asyncio.Queue,
    pending_results: Dict[str, asyncio.Future],
    process_batch_func: Callable[[List[Dict]], List[Dict]],
    cleanup_func: Optional[Callable[[List[Dict], List[Dict]], None]] = None,
    max_batch_size: int = 4,
    batch_timeout_ms: float = 15.0,
) -> asyncio.Task:
    """
    Create and start a batch worker task.

    Args:
        engine_name: Name of the engine (e.g., "parakeet", "whisper")
        transcription_queue: Shared queue for all transcription requests
        pending_results: Dictionary mapping job_id to Future objects
        process_batch_func: Function to process a batch of jobs
        cleanup_func: Optional cleanup function called after each batch
        max_batch_size: Maximum number of requests to batch together
        batch_timeout_ms: Wait up to this many milliseconds to gather a batch

    Returns:
        asyncio.Task for the batch worker

    Example:
        ```python
        def process_parakeet_batch(jobs: List[Dict]) -> List[Dict]:
            # Process jobs with Parakeet model
            results = []
            for job in jobs:
                result = parakeet_model.transcribe(job["audio"])
                results.append({"text": result})
            return results

        worker = create_batch_worker(
            "parakeet",
            queue,
            results_dict,
            process_parakeet_batch
        )
        ```
    """
    worker = BatchWorker(
        engine_name=engine_name,
        transcription_queue=transcription_queue,
        pending_results=pending_results,
        process_batch_func=process_batch_func,
        cleanup_func=cleanup_func,
        max_batch_size=max_batch_size,
        batch_timeout_ms=batch_timeout_ms,
    )

    return asyncio.create_task(worker.run())
