from asyncio import CancelledError
import logging
from uuid import uuid4
from typing import Optional, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, TimeoutError, Future

from langchain_core.messages import AIMessage, ToolCall
from langchain.tools import StructuredTool

from redbox.models.chain import RedboxState
from redbox.api.format import MCPResponseMetadata
from redbox.graph.nodes.runner import exceptions as tool_exceptions
from redbox.graph.nodes.runner.wrap_async import wrap_async_tool

log = logging.getLogger(__name__)


@dataclass
class ToolExecutionResult:
    """Result of parallel tool execution."""

    responses: List[AIMessage]
    failed_tools: List[str]

    @property
    def is_success(self) -> bool:
        return len(self.responses) > 0

    @property
    def is_complete_failure(self) -> bool:
        return len(self.responses) == 0


class ToolRunner:
    """Encapsulates the logic for submitting and parsing individual tool futures."""

    def __init__(
        self,
        tools: list[StructuredTool],
        state: RedboxState,
        max_workers: int,
        is_loop: bool,
        parallel_timeout: float,
    ):
        self.tools = tools
        self.state = state
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.is_loop = is_loop
        self.parallel_timeout = parallel_timeout
        self.log_stub = f"[run_tools_parallel run_id='{str(uuid4())[:8]}']"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)
        return False

    def run(self, tool_calls: list[ToolCall]) -> ToolExecutionResult:
        """Submit all tool calls, collect results, and return aggregated responses or None on total failure."""
        try:
            futures = self._submit_all(tool_calls=tool_calls)
            return self._collect(futures=futures)
        finally:
            self.executor.shutdown(wait=True)

    def _submit_all(self, tool_calls: list[ToolCall]) -> dict[Future, dict]:
        """Submit every tool call to the executor, skipping and logging any that fail to launch."""
        futures = {}

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            try:
                res = self.submit(tool_call=tool_call)
                if res is None:
                    continue
                future, metadata = res
                futures[future] = metadata

            except tool_exceptions.ToolNotFoundError as e:
                log.warning(f"{self.log_stub} Tool '{tool_name}' not found: {e}")

            except tool_exceptions.ToolValidationError as e:
                log.warning(f"{self.log_stub} Tool '{tool_name}' validation error: {e}")

            except tool_exceptions.ToolExecutionError as e:
                log.warning(f"{self.log_stub} Tool '{tool_name}' execution error: {e}")

            except Exception as e:
                log.error(f"{self.log_stub} Unexpected error submitting tool '{tool_name}': {e}", exc_info=True)

        return futures

    def _collect(self, futures: dict[Future, dict]) -> ToolExecutionResult:
        """Wait for all futures, parse results, and return responses or None if everything failed."""
        responses: list[AIMessage] = []
        failed_tools: list[str] = []

        for future in futures.keys():
            future_tool_name = futures[future]["name"]
            try:
                response = self.parse(future=future, metadata=futures[future])
                if response is not None:
                    responses.append(response)

            except tool_exceptions.ToolTimeoutError as e:
                log.warning(f"{self.log_stub} Tool '{future_tool_name}' timed out: {e}")
                failed_tools.append(future_tool_name)

            except tool_exceptions.ToolValidationError as e:
                log.warning(f"{self.log_stub} Tool '{future_tool_name}' validation error: {e}")
                failed_tools.append(future_tool_name)

            except tool_exceptions.ToolExecutionError as e:
                log.warning(f"{self.log_stub} Tool '{future_tool_name}' execution error: {e}")
                failed_tools.append(future_tool_name)

            except Exception as e:
                log.warning(f"{self.log_stub} Tool '{future_tool_name}' error: {e}")
                failed_tools.append(future_tool_name)

        if failed_tools:
            log.error(f"{self.log_stub} {len(failed_tools)} tool(s) failed: {', '.join(failed_tools)}")

        if not responses:
            log.error(
                f"{self.log_stub} Every tool execution has failed or timed out. "
                f"Failed tools: {', '.join(failed_tools) or 'unknown'}."
            )
        else:
            log.warning(
                f"{self.log_stub} Completed. Successful: {len(responses)}, "
                f"Failed: {len(failed_tools)}. Responses: {responses}"
            )

        return ToolExecutionResult(responses=responses, failed_tools=failed_tools)

    def submit(self, tool_call: ToolCall) -> tuple[Future, dict] | None:
        """Find, validate, and submit a tool call to the executor. Returns (future, metadata) or None."""
        tool_name = tool_call.get("name")
        selected_tool: Optional[StructuredTool] = next((tool for tool in self.tools if tool.name == tool_name), None)

        if selected_tool is None:
            available = [tool.name for tool in self.tools]
            raise tool_exceptions.ToolNotFoundError(
                f"Tool '{tool_name}' not found. Available tools: {', '.join(available)}"
            )

        raw_args = tool_call.get("args", {})
        if not isinstance(raw_args, dict):
            raise tool_exceptions.ToolValidationError(
                f"Invalid input for tool '{tool_name}': expected dict, got {type(raw_args).__name__!r}"
            )

        is_intermediate_step = "False"

        try:
            if selected_tool.func and not selected_tool.coroutine:
                args = {**raw_args, "state": self.state}
                future = self.executor.submit(selected_tool.invoke, args)
            else:
                args = {**raw_args}
                if self.is_loop:
                    is_intermediate_step = args.get("is_intermediate_step", "False")
                    log.warning(f"intermediate step: {is_intermediate_step}")
                future = self.executor.submit(wrap_async_tool(selected_tool, tool_name), args)
        except Exception as e:
            raise tool_exceptions.ToolExecutionError(
                f"Failed to submit tool '{tool_name}' for execution: {str(e)}"
            ) from e

        return future, {"name": tool_name, "intermediate_step": is_intermediate_step}

    def parse(self, future: Future, metadata: dict) -> Optional[AIMessage]:
        """Resolve a completed future and transform its result into an AIMessage."""
        future_tool_name = metadata["name"]
        is_intermediate_step = metadata["intermediate_step"]

        try:
            response = future.result(timeout=self.parallel_timeout)
        except TimeoutError as e:
            raise tool_exceptions.ToolTimeoutError(
                f"Tool '{future_tool_name}' timed out after {self.parallel_timeout:.1f}s"
            ) from e
        except (Exception, CancelledError) as e:
            raise tool_exceptions.ToolExecutionError(f"Tool '{future_tool_name}' failed: {str(e)}") from e

        log.warning(f"{self.log_stub} This is what I got from tool '{future_tool_name}': {response}")

        if response is None:
            raise tool_exceptions.ToolExecutionError(
                f"Tool '{future_tool_name}' returned None — may have failed or timed out"
            )

        log.warning(f"{self.log_stub} {future_tool_name} response not None")

        result = response
        if not self.is_loop:
            if isinstance(response, tuple) and isinstance(response[1], MCPResponseMetadata):
                result = response[0]

        else:
            if isinstance(response, tuple) and isinstance(response[1], MCPResponseMetadata):
                res = response[0]
                metadata: MCPResponseMetadata = response[1]
                status = "pass" if res != "" else "fail"
                result = (
                    (
                        res,
                        status,
                        is_intermediate_step,
                        metadata.user_feedback.reason or "Requires feedback from the user.",
                    )
                    if metadata.user_feedback.required
                    else (res, status, is_intermediate_step)
                )

        raw_res = result[0] if isinstance(result, tuple) else result

        if raw_res is None:
            raise tool_exceptions.ToolValidationError(f"Tool '{future_tool_name}' returned None")

        if not isinstance(raw_res, str):
            raise tool_exceptions.ToolValidationError(
                f"Tool '{future_tool_name}' returned non-string type: {type(raw_res).__name__}"
            )

        if not raw_res.strip():
            raise tool_exceptions.ToolValidationError(
                f"Tool '{future_tool_name}' returned empty or whitespace-only response"
            )

        return AIMessage(result)
