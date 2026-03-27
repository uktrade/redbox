import logging
import threading
from uuid import uuid4
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed, Future

from langchain_core.messages import AIMessage, ToolCall
from langchain.tools import StructuredTool

from redbox.models.chain import RedboxState
from redbox.api.format import MCPResponseMetadata
from redbox.graph.nodes.runner import exceptions as tool_exceptions
from redbox.graph.nodes.runner.wrap_async import wrap_async_tool

log = logging.getLogger(__name__)


def run_with_timeout(func, args, timeout):
    """Run a a function with a timeout and return its result or None if it times out or fails.
    This function can be used to set a timeout for tool execution"""
    result = [None]
    exception = [None]
    completed = [False]

    def target():
        try:
            result[0] = func(args)
        except Exception as e:
            exception[0] = e
        finally:
            completed[0] = True

    thread = threading.Thread(target=target)
    thread.daemon = True  # The thread will exit when the main program exits
    thread.start()
    thread.join(timeout)  # applying timeout constraint

    if not completed[0]:  # if it times out
        log.warning(f"Tool execution timed out after {timeout} seconds")
        return None
    if exception[0]:  # if the tool fails
        log.warning(f"Tool execution failed: {str(exception[0])}")
        return None
    return result[0]


class ToolRunner:
    """Encapsulates the logic for submitting and parsing individual tool futures."""

    def __init__(
        self,
        tools: list[StructuredTool],
        state: RedboxState,
        executor: ThreadPoolExecutor,
        is_loop: bool,
        parallel_timeout: float,
    ):
        self.tools = tools
        self.state = state
        self.executor = executor
        self.is_loop = is_loop
        self.parallel_timeout = parallel_timeout
        self.log_stub = f"[run_tools_parallel run_id='{str(uuid4())[:8]}']"

    def run(self, tool_calls: list[ToolCall]) -> list[AIMessage] | None:
        """Submit all tool calls, collect results, and return aggregated responses or None on total failure."""
        futures = self._submit_all(tool_calls=tool_calls)
        return self._collect(futures=futures)

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
                log.warning(f"{self.log_stub} '{tool_name}' not found: {e}")
            except tool_exceptions.ToolInputValidationError as e:
                log.warning(f"{self.log_stub} '{tool_name}' invalid input: {e}")
            except tool_exceptions.BaseToolRunnerException as e:
                log.warning(f"{self.log_stub} '{tool_name}' failed to submit: {e}")
        return futures

    def _collect(self, futures: dict[Future, dict]) -> list[AIMessage] | None:
        """Wait for all futures, parse results, and return responses or None if everything failed."""
        responses = []
        failed_tools: list[str] = []

        for future in as_completed(futures.keys(), timeout=self.parallel_timeout):
            future_tool_name = futures[future]["name"]
            try:
                response = self.parse(future=future, metadata=futures[future])
                if response is not None:
                    responses.append(response)
            except tool_exceptions.ToolTimeoutError as e:
                log.warning(f"{self.log_stub} '{future_tool_name}' timed out after {e.timeout_seconds:.1f}s.")
                failed_tools.append(future_tool_name)
            except tool_exceptions.ToolOutputValidationError as e:
                log.warning(f"{self.log_stub} '{future_tool_name}' returned an invalid output: {e}")
                failed_tools.append(future_tool_name)
            except tool_exceptions.ToolFailedError as e:
                log.warning(f"{self.log_stub} '{future_tool_name}' failed: {e}")
                if e.cause:
                    log.debug(f"{self.log_stub} '{future_tool_name}' cause: {e.cause}", exc_info=e.cause)
                failed_tools.append(future_tool_name)
            except tool_exceptions.BaseToolRunnerException as e:
                log.warning(f"{self.log_stub} '{future_tool_name}' tool runner error: {e}")
                failed_tools.append(future_tool_name)

        if failed_tools:
            log.warning(f"{self.log_stub} {len(failed_tools)} tool(s) failed: {', '.join(failed_tools)}")

        if not responses:
            log.warning(
                f"{self.log_stub} Every tool execution has failed or timed out. "
                f"Failed tools: {', '.join(failed_tools) or 'unknown'}."
            )
            return None

        log.warning(
            f"{self.log_stub} Completed. Successful: {len(responses)}, "
            f"Failed: {len(failed_tools)}. Responses: {responses}"
        )
        return responses

    def submit(self, tool_call: ToolCall) -> tuple[Future, dict] | None:
        """Find, validate, and submit a tool call to the executor. Returns (future, metadata) or None."""
        tool_name = tool_call.get("name")
        selected_tool: Optional[StructuredTool] = next((tool for tool in self.tools if tool.name == tool_name), None)

        if selected_tool is None:
            raise tool_exceptions.ToolNotFoundError(
                tool_name=tool_name, available_tools=[tool.name for tool in self.tools]
            )

        raw_args = tool_call.get("args", {})
        if not isinstance(raw_args, dict):
            raise tool_exceptions.ToolInputValidationError(
                tool_name=tool_name,
                field="args",
                value=raw_args,
                reason=f"expected dict, got {type(raw_args).__name__!r}",
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
            raise tool_exceptions.ToolFailedError(tool_name=tool_name, cause=e) from e

        return future, {"name": tool_name, "intermediate_step": is_intermediate_step}

    def parse(self, future: Future, metadata: dict) -> Optional[AIMessage]:
        """Resolve a completed future and transform its result into an AIMessage."""
        future_tool_name = metadata["name"]
        is_intermediate_step = metadata["intermediate_step"]

        try:
            response = future.result()
        except TimeoutError as e:
            raise tool_exceptions.ToolTimeoutError(future_tool_name, timeout_seconds=self.parallel_timeout) from e
        except Exception as e:
            raise tool_exceptions.ToolFailedError(future_tool_name, cause=e) from e

        log.warning(f"{self.log_stub} This is what I got from tool '{future_tool_name}': {response}")

        if response is None:
            raise tool_exceptions.ToolFailedError(
                future_tool_name, stderr="Tool returned None — may have failed or timed out"
            )

        log.warning(f"{self.log_stub} {future_tool_name} response not None")

        result = None

        if not self.is_loop:
            if isinstance(response, tuple):
                if isinstance(response[1], MCPResponseMetadata):
                    result = response[0]
                else:
                    result = response

        else:
            if isinstance(response, tuple):
                if isinstance(response[1], MCPResponseMetadata):
                    res = response[0]
                    metadata = response[1]
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
                else:
                    result = response
            else:
                result = response

        # if (not self.is_loop and isinstance(response, str)) or (self.is_loop and isinstance(response, tuple)):
        #     log.warning(f"{self.log_stub} {future_tool_name} my non-transformed response: {response}")
        #     result = response

        # elif self.is_loop and isinstance(response, str):
        #     try:
        #         result_dict = json.loads(response)
        #         is_empty = result_dict.get("total") == 0
        #         log.warning(f"{self.log_stub} {future_tool_name} is_empty {is_empty}")
        #     except json.JSONDecodeError:
        #         is_empty = response in ["", "None", "[]"]

        #     status = "fail" if is_empty else "pass"

        #     if is_empty:
        #         log.warning(f"No records returned from {future_tool_name} tool")
        #         response = "Error message: Empty response"

        #     transformed_response = (response, status, is_intermediate_step)
        #     log.warning("my transformed response")
        #     log.warning(transformed_response)
        #     result = transformed_response

        # else:
        #     raise tool_exceptions.ToolOutputValidationError(
        #         future_tool_name,
        #         reason=(
        #             f"expected str (is_loop=False) or tuple (is_loop=True), "
        #             f"got {type(response).__name__!r} — is_loop={self.is_loop}"
        #         ),
        #     )

        raw_res = result
        if isinstance(raw_res, tuple):
            raw_res = raw_res[0]

        if not raw_res or not isinstance(raw_res, str) or not raw_res.strip():
            raise tool_exceptions.ToolOutputValidationError(
                future_tool_name,
                reason=f"empty or whitespace-only response body: {repr(raw_res)}",
            )

        return AIMessage(result)
