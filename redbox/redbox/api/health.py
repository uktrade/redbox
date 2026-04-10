import asyncio
import aiohttp
import logging

logger = logging.getLogger(__name__)


async def wait_for_health(
    url: str,
    headers: dict,
    success_codes: list[int],
    wait_seconds: float = 60.0,
    interval: float = 2.0,
    request_timeout: float = 5.0,
) -> bool:
    """
    Polls URL for a response

    Parameters:
        url: str - the endpoint to poll
        headers: dict - headers to embed in the endpoint call
        success_codes: list[int] - successful response codes ie. [200]
        wait_seconds: float - the deadline for polling
        interval: float - the interval for polling
        request_timeout: float - the timeout for the request

    Outputs:
        bool - True if successful response, False otherwise
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + wait_seconds
    attempt = 0

    while loop.time() < deadline:
        attempt += 1
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=request_timeout),
                    headers=headers,
                ) as response,
            ):
                if response.status in success_codes:
                    logger.info("Health check passed after %d attempt(s)", attempt)
                    return True
                logger.debug("Health check attempt %d: status %d", attempt, response.status)
        except (
            aiohttp.ClientConnectionError,  # connection refused, DNS failure
            aiohttp.ClientResponseError,  # bad HTTP response
            aiohttp.ServerTimeoutError,  # per-request timeout
            TimeoutError,  # underlying socket timeout
        ) as e:
            logger.debug("Health check attempt %d failed: %s", attempt, e)

        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        await asyncio.sleep(min(interval, remaining))

    logger.warning("Health check timed out after %d attempt(s): %s", attempt, url)
    return False
