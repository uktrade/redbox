from asyncio import CancelledError
import logging
from uuid import uuid4
from typing import Optional, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError, Future

from langchain_core.messages import AIMessage, ToolCall
from langchain.tools import StructuredTool

from redbox.models.chain import RedboxState
from redbox.api.format import MCPResponseMetadata
from redbox.graph.nodes.runner import exceptions as tool_exceptions
from redbox.graph.nodes.runner.wrap_async import execute_mcp_tools
import redbox.graph.nodes.runner.models as tr_models

log = logging.getLogger(__name__)


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

    def run(self, tool_calls: list[ToolCall]) -> tr_models.Result:
        """Submit all tool calls, collect results, and return aggregated responses or None on total failure."""
        try:
            futures, failures = self._submit_all(tool_calls=tool_calls)
            return self._collect(futures=futures, failures=failures)
        finally:
            self.executor.shutdown(wait=True)

    def group_tool_calls(
        self, tool_calls: list[ToolCall]
    ) -> Tuple[tr_models.ToolCallWrapper.Sync, tr_models.ToolCallWrapper.MCPAsync]:
        sync_tools: list[tr_models.ToolCallWrapper.Sync] = []
        mcp_async_tools: dict[str, tr_models.ToolCallWrapper.MCPAsync] = {}

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            selected_tool: Optional[StructuredTool] = next(
                (tool for tool in self.tools if tool.name == tool_name), None
            )

            if selected_tool.func and not selected_tool.coroutine:
                sync_tools.append(tr_models.ToolCallWrapper.Sync(tool_call=tool_call))
            else:
                mcp_url = selected_tool.metadata["url"]
                creator_type = selected_tool.metadata["creator_type"]
                sso_access_token = selected_tool.metadata["sso_access_token"]

                if mcp_url:
                    if mcp_url in mcp_async_tools.keys():
                        mcp_async_tools[mcp_url].tool_calls.append(tool_call)
                    else:
                        mcp_async_tools[mcp_url] = tr_models.ToolCallWrapper.MCPAsync(
                            mcp_server=mcp_url,
                            access_token=sso_access_token,
                            creator_type=creator_type,
                            tool_calls=[tool_call],
                        )

        return sync_tools, mcp_async_tools.values()

    def _submit_all(
        self, tool_calls: list[ToolCall]
    ) -> tuple[dict[Future, dict], list[tr_models.ToolCallResult.Failure]]:
        """Submit every tool call to the executor, skipping and logging any that fail to launch."""
        sync_calls, async_mcp_calls = self.group_tool_calls(tool_calls=tool_calls)

        futures = {}
        failures: list[tr_models.ToolCallResult.Failure] = []

        for sync_call in sync_calls:
            tool_call = sync_call.tool_call
            tool_name = tool_call.get("name")
            raw_args = tool_call.get("args", {})
            raw_args.pop("name", None)

            try:
                res = self.submit_sync(tool_call=tool_call)
                future, metadata = res
                futures[future] = metadata

            except (
                tool_exceptions.ToolTimeoutError,
                tool_exceptions.ToolValidationError,
                tool_exceptions.ToolExecutionError,
                tool_exceptions.ToolNotFoundError,
            ) as e:
                log.warning(f"{self.log_stub} {e}")
                failures.append(
                    tr_models.ToolCallResult.Failure(
                        tool_name=tool_name, error=str(e), metadata={"tool_args": raw_args}
                    )
                )

            except Exception as e:
                err = f"Unexpected error submitting tool '{tool_name}': {e}"
                log.error(f"{self.log_stub} {err}", exc_info=True)
                failures.append(
                    tr_models.ToolCallResult.Failure(tool_name=tool_name, error=err, metadata={"tool_args": raw_args})
                )

        for async_mcp_call in async_mcp_calls:
            try:
                res = self.submit_mcp_async(mcp_server=async_mcp_call.mcp_server, tool_call=async_mcp_call)
                future, metadata = res
                futures[future] = metadata

            except (
                tool_exceptions.ToolTimeoutError,
                tool_exceptions.ToolValidationError,
                tool_exceptions.ToolExecutionError,
                tool_exceptions.ToolNotFoundError,
            ) as e:
                log.warning(f"{self.log_stub} {e}")
                failures.append(
                    tr_models.ToolCallResult.Failure(
                        tool_name=async_mcp_call.mcp_server, error=str(e), metadata={"tool_args": {}}
                    )
                )

            except Exception as e:
                err = f"Unexpected error submitting Async MCP Server tool calls '{async_mcp_call.mcp_server}': {e}"
                log.error(f"{self.log_stub} {err}", exc_info=True)
                failures.append(
                    tr_models.ToolCallResult.Failure(
                        tool_name=async_mcp_call.mcp_server, error=err, metadata={"tool_args": {}}
                    )
                )

        return futures, failures

    def _collect(
        self, futures: dict[Future, dict], failures: list[tr_models.ToolCallResult.Failure]
    ) -> tr_models.Result:
        """Wait for all futures, parse results, and return responses or None if everything failed."""
        results: List[tr_models.ToolCallResult.Success] = []

        for future in futures.keys():
            future_tool_name = futures[future]["name"]
            try:
                metadata = dict(futures[future])
                metadata.pop("name", None)

                response = self.parse(future=future, metadata=futures[future])
                if response is not None:
                    if isinstance(response, list):
                        for item in response:
                            results.append(
                                tr_models.ToolCallResult.Success(
                                    tool_name=future_tool_name, response=item, metadata=metadata
                                )
                            )
                    else:
                        results.append(
                            tr_models.ToolCallResult.Success(
                                tool_name=future_tool_name, response=response, metadata=metadata
                            )
                        )

            except (
                tool_exceptions.ToolTimeoutError,
                tool_exceptions.ToolValidationError,
                tool_exceptions.ToolExecutionError,
                tool_exceptions.ToolNotFoundError,
            ) as e:
                log.warning(f"{self.log_stub} {e}")
                failures.append(
                    tr_models.ToolCallResult.Failure(tool_name=future_tool_name, error=str(e), metadata=metadata)
                )

            except Exception as e:
                err = f"Tool '{future_tool_name}' error: {e}"
                log.warning(f"{self.log_stub} {err}")
                failures.append(
                    tr_models.ToolCallResult.Failure(tool_name=future_tool_name, error=err, metadata=metadata)
                )

        failed_tools = [f"{fr.tool_name} - {fr.error}" for fr in failures]
        if failed_tools:
            log.error(f"{self.log_stub} {len(failed_tools)} tool(s) failed: {', '.join(failed_tools)}")

        if not results:
            log.error(
                f"{self.log_stub} Every tool execution has failed or timed out. "
                f"Failed tools: {', '.join(failed_tools) or 'unknown'}."
            )
        else:
            log.warning(
                f"{self.log_stub} Completed. Successful: {len(results)}, Failed: {len(failures)}. Responses: {results}"
            )

        return tr_models.Result(results=results, failures=failures)

    def validate(self, tool_call: ToolCall) -> tuple[str, StructuredTool, dict]:
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

        return tool_name, selected_tool, raw_args

    def submit_sync(self, tool_call: tr_models.ToolCallWrapper.Sync) -> tuple[Future, dict]:
        """Find, validate, and submit a tool call to the executor. Returns (future, metadata)"""
        tool_name, selected_tool, raw_args = self.validate(tool_call=tool_call)
        is_intermediate_step = "False"

        try:
            args = {**raw_args, "state": self.state}
            future = self.executor.submit(selected_tool.invoke, args)
        except Exception as e:
            raise tool_exceptions.ToolExecutionError(
                f"Failed to submit tool '{tool_name}' for execution: {str(e)}"
            ) from e

        return future, {
            "name": tool_name,
            "intermediate_step": is_intermediate_step,
            "tool_args": raw_args,
            "future_result_type": tool_call.future_result_type,
        }

    # def submit_async(self, tool_call: RunnerToolCall.Sync) -> tuple[Future, dict]:
    #     """Find, validate, and submit a tool call to the executor. Returns (future, metadata)"""
    #     tool_name, selected_tool, raw_args = self.validate(tool_call=tool_call)
    #     is_intermediate_step = "False"

    #     try:
    #         # if selected_tool.func and not selected_tool.coroutine:
    #         args = {**raw_args, "state": self.state}
    #         future = self.executor.submit(selected_tool.invoke, args)
    #         # else:
    #         #     args = {**raw_args}
    #         #     if self.is_loop:
    #         #         is_intermediate_step = args.get("is_intermediate_step", "False")
    #         #         log.warning(f"intermediate step: {is_intermediate_step}")
    #         #     future = self.executor.submit(wrap_async_tool(selected_tool, tool_name), args)
    #         #     future_type = FutureResultType.ASYNC
    #     except Exception as e:
    #         raise tool_exceptions.ToolExecutionError(
    #             f"Failed to submit tool '{tool_name}' for execution: {str(e)}"
    #         ) from e

    #     return future, {
    #         "name": tool_name,
    #         "intermediate_step": is_intermediate_step,
    #         "tool_args": raw_args,
    #         "future_result_type": tool_call.future_result_type,
    #     }

    def submit_mcp_async(self, mcp_server: str, tool_call: tr_models.ToolCallWrapper.MCPAsync) -> tuple[Future, dict]:
        """Find, validate, and submit a tool call to the executor. Returns (future, metadata)"""
        try:
            future = self.executor.submit(execute_mcp_tools(mcp_input=tool_call))
        except Exception as e:
            raise tool_exceptions.ToolExecutionError(
                f"Failed to submit tool 'MCP_{mcp_server}' for execution: {str(e)}"
            ) from e

        return future, {
            "name": f"MCP_{mcp_server}",
            "intermediate_step": "False",
            "tool_args": {},
            "future_result_type": tool_call.future_result_type,
        }

    def parse(self, future: Future, metadata: dict) -> AIMessage | list[AIMessage]:
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

        if isinstance(response, list):
            results = []
            for item in response:
                results.append(
                    self.parse_response(
                        future_tool_name=future_tool_name, response=item, is_intermediate_step=is_intermediate_step
                    )
                )

            return results

        return self.parse_response(
            future_tool_name=future_tool_name, response=response, is_intermediate_step=is_intermediate_step
        )

    def parse_response(self, future_tool_name: str, response: Any, is_intermediate_step: str) -> AIMessage:
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
