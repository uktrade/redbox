from asyncio import CancelledError
import logging
from uuid import uuid4
from typing import Optional, List, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError

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
            submitted_request = self._submit_all(tool_calls=tool_calls)
            return self._collect(submitted_request=submitted_request)
        finally:
            self.executor.shutdown(wait=True)

    def parse_tool_calls(self, tool_calls: list[ToolCall]) -> tr_models.ParsedRunRequest:
        failures: list[tr_models.ToolCallResult.Failure] = []
        sync_calls: list[tr_models.ToolCallRequest.Sync] = []
        mcp_async_server_calls: dict[str, tr_models.ToolCallRequest.MCPAsync] = {}

        for tool_call in tool_calls:
            try:
                tool_name, selected_tool, _ = self.validate_tool_call(tool_call=tool_call)

                if selected_tool.func and not selected_tool.coroutine:
                    sync_calls.append(tr_models.ToolCallRequest.Sync(tool_call=tool_call))

                else:
                    mcp_url = selected_tool.metadata["url"]
                    creator_type = selected_tool.metadata["creator_type"]
                    sso_access_token = selected_tool.metadata["sso_access_token"]

                    if mcp_url:
                        if mcp_url in mcp_async_server_calls.keys():
                            mcp_async_server_calls[mcp_url].tool_calls.append(tool_call)
                        else:
                            mcp_async_server_calls[mcp_url] = tr_models.ToolCallRequest.MCPAsync(
                                mcp_server=mcp_url,
                                access_token=sso_access_token,
                                creator_type=creator_type,
                                tool_calls=[tool_call],
                            )

                    else:
                        failures.append(
                            tr_models.ToolCallResult.Failure(
                                tool_name=tool_name,
                                metadata={"tool_args": tool_call.get("args")},
                                error=f"Unrecognised tool configuration for '{tool_name}'.",
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
                    tr_models.ToolCallResult.Failure(
                        tool_name=tool_call.get("name"), error=str(e), metadata={"tool_args": tool_call.get("args")}
                    )
                )

        return tr_models.ParsedRunRequest(
            calls=tr_models.RunRequestCalls(sync=sync_calls, mcp_async=mcp_async_server_calls.values()),
            failures=failures,
        )

    def _submit_all(self, tool_calls: list[ToolCall]) -> tr_models.SubmittedRunRequest:
        """Submit every tool call to the executor, skipping and logging any that fail to launch."""
        run_request = self.parse_tool_calls(tool_calls=tool_calls)

        futures: list[tr_models.SubmittedToolCallRequest] = []
        failures: list[tr_models.ToolCallResult.Failure] = run_request.failures

        for sync_call in run_request.calls.sync:
            tool_name = sync_call.tool_call.get("name")
            raw_args = sync_call.tool_call.get("args", {})
            raw_args.pop("name", None)

            try:
                res = self.submit_sync(sync_call=sync_call)
                futures.append(res)

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

        for mcp_async_call in run_request.calls.mcp_async:
            try:
                res = self.submit_mcp_async(mcp_async_call=mcp_async_call)
                futures.append(res)

            except (
                tool_exceptions.ToolTimeoutError,
                tool_exceptions.ToolValidationError,
                tool_exceptions.ToolExecutionError,
                tool_exceptions.ToolNotFoundError,
            ) as e:
                log.warning(f"{self.log_stub} {e}")
                failures.append(
                    tr_models.ToolCallResult.Failure(
                        tool_name=f"MCP_async__{mcp_async_call.creator_type}__{mcp_async_call.mcp_server}",
                        error=str(e),
                        metadata={
                            "mcp_server": mcp_async_call.mcp_server,
                            "creator_type": mcp_async_call.creator_type,
                            "tool_calls": [
                                {"name": tc.get("name"), "args": tc.get("args")} for tc in mcp_async_call.tool_calls
                            ],
                        },
                    )
                )

            except Exception as e:
                err = f"Unexpected error submitting Async MCP Server tool calls '{mcp_async_call.mcp_server}': {e}"
                log.error(f"{self.log_stub} {err}", exc_info=True)
                failures.append(
                    tr_models.ToolCallResult.Failure(
                        tool_name=f"MCP_async__{mcp_async_call.creator_type}__{mcp_async_call.mcp_server}",
                        error=err,
                        metadata={
                            "mcp_server": mcp_async_call.mcp_server,
                            "creator_type": mcp_async_call.creator_type,
                            "tool_calls": [
                                {"name": tc.get("name"), "args": tc.get("args")} for tc in mcp_async_call.tool_calls
                            ],
                        },
                    )
                )

        return tr_models.SubmittedRunRequest(futures=futures, failures=failures)

    def _collect(self, submitted_request: tr_models.SubmittedRunRequest) -> tr_models.Result:
        """Wait for all futures, parse results, and return responses or None if everything failed."""
        results: List[tr_models.ToolCallResult.Success] = []
        failures: List[tr_models.ToolCallResult.Failure] = submitted_request.failures

        for request in submitted_request.futures:
            try:
                response = self.execute_request(request=request)
                if response is not None:
                    if isinstance(response, list):
                        for item in response:
                            results.append(
                                tr_models.ToolCallResult.Success(
                                    tool_name=request.name,
                                    response=item,
                                    metadata={**request.metadata, "tool_args": request.future_args},
                                )
                            )
                    else:
                        results.append(
                            tr_models.ToolCallResult.Success(
                                tool_name=request.name,
                                response=response,
                                metadata={**request.metadata, "tool_args": request.future_args},
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
                    tr_models.ToolCallResult.Failure(
                        tool_name=request.name,
                        error=str(e),
                        metadata={**request.metadata, "tool_args": request.future_args},
                    )
                )

            except Exception as e:
                err = f"Tool '{request.name}' error: {e}"
                log.warning(f"{self.log_stub} {err}")
                failures.append(
                    tr_models.ToolCallResult.Failure(
                        tool_name=request.name,
                        error=err,
                        metadata={**request.metadata, "tool_args": request.future_args},
                    )
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

    def validate_tool_call(self, tool_call: ToolCall) -> tuple[str, StructuredTool, dict]:
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

    def submit_sync(self, sync_call: tr_models.ToolCallRequest.Sync) -> tr_models.SubmittedToolCallRequest:
        """Find, validate, and submit a sync tool call to the executor."""
        tool_name, selected_tool, tool_args = self.validate_tool_call(tool_call=sync_call.tool_call)
        is_intermediate_step = "False"

        try:
            args = {**tool_args, "state": self.state}
            future = self.executor.submit(selected_tool.invoke, args)
        except Exception as e:
            raise tool_exceptions.ToolExecutionError(
                f"Failed to submit tool '{tool_name}' for execution: {str(e)}"
            ) from e

        return tr_models.SubmittedToolCallRequest(
            name=tool_name,
            result_type=sync_call.future_result_type,
            future=future,
            future_args=tool_args,
            metadata={"intermediate_step": is_intermediate_step},
        )

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

    def submit_mcp_async(
        self, mcp_async_call: tr_models.ToolCallRequest.MCPAsync
    ) -> tr_models.SubmittedToolCallRequest:
        """Find, validate, and submit a grouped MCP async tool call to the executor. Returns (future, metadata)"""
        is_intermediate_step = "False"

        try:
            future = self.executor.submit(execute_mcp_tools(mcp_input=mcp_async_call))
        except Exception as e:
            raise tool_exceptions.ToolExecutionError(
                f"Failed to submit MCP async calls to server '{mcp_async_call.mcp_server}' with creator_type '{mcp_async_call.creator_type}' for execution: {str(e)}"
            ) from e

        return tr_models.SubmittedToolCallRequest(
            name=f"MCP_async__{mcp_async_call.creator_type}__{mcp_async_call.mcp_server}",
            result_type=mcp_async_call.future_result_type,
            future=future,
            future_args={
                "mcp_server": mcp_async_call.mcp_server,
                "creator_type": mcp_async_call.creator_type,
                "tool_calls": [{"name": tc.get("name"), "args": tc.get("args")} for tc in mcp_async_call.tool_calls],
            },
            metadata={"intermediate_step": is_intermediate_step},
        )

    def execute_request(self, request: tr_models.SubmittedToolCallRequest) -> AIMessage | list[AIMessage]:
        """Resolve a completed future and transform its result into an AIMessage."""
        future_tool_name = request.name
        is_intermediate_step = request.metadata["intermediate_step"]

        try:
            response = request.future.result(timeout=self.parallel_timeout)
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
