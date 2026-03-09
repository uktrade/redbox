import logging
import json
from uuid import uuid4
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed, Future

from langchain_core.messages import AIMessage, ToolCall

import redbox.graph.nodes.runner.exceptions as tool_exceptions
from redbox.graph.nodes.runner.async_util import wrap_async_tool

log = logging.getLogger(__name__)


class ToolRunner:
    """Encapsulates the logic for submitting and parsing individual tool futures."""

    def __init__(self, tools, state, executor: ThreadPoolExecutor, is_loop: bool, parallel_timeout: float):
        self.tools = tools
        self.state = state
        self.executor = executor
        self.is_loop = is_loop
        self.parallel_timeout = parallel_timeout
        self.log_stub = f"[run_tools_parallel run_id='{str(uuid4())[:8]}']"

    def run(self, tool_calls: list[ToolCall]) -> list[AIMessage] | None:
        """Submit all tool calls, collect results, and return aggregated responses or None on total failure."""
        futures = self._submit_all(tool_calls)
        return self._collect(futures)

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
            except tool_exceptions.ToolRunnerError as e:
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
            except tool_exceptions.ToolRunnerError as e:
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
        selected_tool = next((tool for tool in self.tools if tool.name == tool_name), None)

        if selected_tool is None:
            raise tool_exceptions.ToolNotFoundError(
                tool_name=tool_name, available_tools=[tool.name for tool in self.tools]
            )

        args = tool_call.get("args", {})
        if not isinstance(args, dict):
            raise tool_exceptions.ToolInputValidationError(
                tool_name=tool_name,
                field="args",
                value=args,
                reason=f"expected dict, got {type(args).__name__!r}",
            )

        log.warning(f"args: {args}")
        is_intermediate_step = "False"

        try:
            if selected_tool.func and not selected_tool.coroutine:
                args["state"] = self.state
                future = self.executor.submit(selected_tool.invoke, args)
            else:
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

        if (not self.is_loop and isinstance(response, str)) or (self.is_loop and isinstance(response, tuple)):
            log.warning(f"{self.log_stub} {future_tool_name} my non-transformed response: {response}")
            result = response

        elif self.is_loop and isinstance(response, str):
            try:
                result_dict = json.loads(response)
                is_empty = result_dict.get("total") == 0
                log.warning(f"{self.log_stub} {future_tool_name} is_empty {is_empty}")
            except json.JSONDecodeError:
                is_empty = response in ["", "None", "[]"]

            status = "fail" if is_empty else "pass"

            if is_empty:
                log.warning(f"No records returned from {future_tool_name} tool")
                response = "Error message: Empty response"

            transformed_response = (response, status, is_intermediate_step)
            log.warning("my transformed response")
            log.warning(transformed_response)
            result = transformed_response

        else:
            raise tool_exceptions.ToolOutputValidationError(
                future_tool_name,
                reason=(
                    f"expected str (is_loop=False) or tuple (is_loop=True), "
                    f"got {type(response).__name__!r} — is_loop={self.is_loop}"
                ),
            )

        raw_res = response
        if isinstance(raw_res, tuple):
            raw_res = raw_res[0]

        if not raw_res or not isinstance(raw_res, str) or not raw_res.strip():
            raise tool_exceptions.ToolOutputValidationError(
                future_tool_name,
                reason=f"empty or whitespace-only response body: {repr(raw_res)}",
            )

        return AIMessage(result)


# def run_tools_parallel(ai_msg, tools, state, parallel_timeout=60, is_loop=False):
#     log.warning(f"{log_stub} Starting tool execution.")

#     if not ai_msg.tool_calls:
#         log.warning(f"{log_stub} No tool calls detected. Returning agent content.")
#         return ai_msg.content

#     log.warning(
#         f"{log_stub} {len(ai_msg.tool_calls)} tool call(s) detected: {[tc.get('name') for tc in ai_msg.tool_calls]}"
#     )

#     max_workers = min(10, len(ai_msg.tool_calls))
#     log.warning(f"{log_stub} Creating ThreadPoolExecutor(max_workers={max_workers})")

#     futures = {}
#     try:
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             runner = ToolRunner(
#                 tools=tools,
#                 state=state,
#                 executor=executor,
#                 is_loop=is_loop,
#                 parallel_timeout=parallel_timeout,
#                 log_stub=log_stub,
#             )

#             for tool_call in ai_msg.tool_calls:
#                 tool_name = tool_call.get("name")
#                 try:
#                     res = runner.submit(tool_call=tool_call)
#                     if res is None:
#                         continue
#                     future, metadata = res
#                     futures[future] = metadata

#                 except tool_exceptions.ToolNotFoundError as e:
#                     log.warning(f"{log_stub} '{tool_name}' not found: {e}")
#                 except tool_exceptions.ToolInputValidationError as e:
#                     log.warning(f"{log_stub} '{tool_name}' invalid input: {e}")
#                 except tool_exceptions.ToolRunnerError as e:
#                     log.warning(f"{log_stub} '{tool_name}' failed to submit: {e}")

#             responses = []
#             failed_tools: list[str] = []
#             for future in as_completed(futures.keys(), timeout=parallel_timeout):
#                 future_tool_name = futures[future]["name"]
#                 try:
#                     response = runner.parse(future=future, metadata=futures[future])
#                     if response is not None:
#                         responses.append(response)

#                 except tool_exceptions.ToolTimeoutError as e:
#                     log.warning(f"{log_stub} '{future_tool_name}' timed out after {e.timeout_seconds:.1f}s.")
#                     failed_tools.append(future_tool_name)
#                 except tool_exceptions.ToolOutputValidationError as e:
#                     log.warning(f"{log_stub} '{future_tool_name}' returned an invalid output: {e}")
#                     failed_tools.append(future_tool_name)
#                 except tool_exceptions.ToolFailedError as e:
#                     log.warning(f"{log_stub} '{future_tool_name}' failed: {e}")
#                     if e.cause:
#                         log.debug(f"{log_stub} '{future_tool_name}' cause: {e.cause}", exc_info=e.cause)
#                     failed_tools.append(future_tool_name)
#                 except tool_exceptions.ToolRunnerError as e:
#                     log.warning(f"{log_stub} '{future_tool_name}' tool runner error: {e}")
#                     failed_tools.append(future_tool_name)

#             if failed_tools:
#                 log.warning(f"{log_stub} {len(failed_tools)} tool(s) failed: {', '.join(failed_tools)}")

#             if not responses:
#                 log.warning(
#                     f"{log_stub} Every tool execution has failed or timed out. "
#                     f"Failed tools: {', '.join(failed_tools) or 'unknown'}."
#                 )
#                 return None

#             log.warning(
#                 f"{log_stub} Completed. Successful: {len(responses)}, "
#                 f"Failed: {len(failed_tools)}. Responses: {responses}"
#             )
#             return responses

#     except TimeoutError:
#         log.warning(f"{log_stub} Global parallel tool execution timed out after {parallel_timeout} seconds.")
#         return None
#     except Exception as e:
#         log.warning(f"{log_stub} Unexpected error in parallel tool execution: {str(e)}", exc_info=True)
#         return None
