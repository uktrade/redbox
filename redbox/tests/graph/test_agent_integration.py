from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage

from redbox import Redbox
from redbox.models.chain import (
    RedboxQuery,
    RedboxState,
    StructuredResponseWithCitations,
    configure_agent_task_plan,
)
from redbox.models.chat import ChatRoute
from redbox.models.settings import Settings
from redbox.test.data import (
    GenericFakeChatModelWithTools,
    RedboxTestData,
    generate_test_cases,
    mock_all_chunks_retriever,
    mock_metadata_retriever,
    mock_parameterised_retriever,
)

EVALUATOR_ANSWER = AIMessage(
    content=StructuredResponseWithCitations(answer="Here is the answer summary.", citations=[]).model_dump_json()
)


@pytest.mark.asyncio
async def test_summarisation_agent_returns_response(mocker):
    """
    Integration test to check Summarisation_Agent produces a non blank reponse when asked to summarise a file that has been input
    """
    doc_count, tokens = 1, 5000
    test_case = generate_test_cases(
        query=RedboxQuery(
            question="Give me a summary of this document.",
            s3_keys=["test-file.txt"],
            user_uuid=uuid4(),
            chat_history=[],
            permitted_s3_keys=[],
        ),
        test_data=[
            RedboxTestData(
                number_of_docs=doc_count,
                tokens_in_all_docs=tokens,
                llm_responses=["Here is the answer summary."],
                expected_route=ChatRoute.newroute,
            )
        ],
        test_id="summarisation-test",
    )[0]

    # Wiring the planner to use the Summarisation_Agent
    agent_task, multi_agent_plan = configure_agent_task_plan({"Summarisation_Agent": "Summarisation_Agent"})
    plan = multi_agent_plan().model_copy(update={"tasks": [agent_task()]})

    planner_response = GenericFakeChatModelWithTools(messages=iter([plan.model_dump_json()]))
    planner_response._default_config = {"model": "bedrock"}
    worker_response = GenericFakeChatModelWithTools(messages=iter([AIMessage(content="Here is the answer summary.")]))
    worker_response
    evaluator_response = GenericFakeChatModelWithTools(messages=iter([EVALUATOR_ANSWER]))
    evaluator_response._default_config = {"model": "bedrock"}

    mocker.patch("redbox.chains.runnables.get_chat_llm", side_effect=[planner_response, worker_response])
    mocker.patch("redbox.graph.nodes.processes.get_chat_llm", return_value=evaluator_response)

    app = Redbox(
        all_chunks_retriever=mock_all_chunks_retriever(test_case.docs),
        parameterised_retriever=mock_parameterised_retriever(test_case.docs),
        metadata_retriever=mock_metadata_retriever(test_case.docs),
        env=Settings(),
        debug=True,
    )

    token_events = []

    async def collect_tokens(t):
        token_events.append(t)

    final_state = await app.run(
        input=RedboxState(request=test_case.query),
        response_tokens_callback=collect_tokens,
    )

    # Assertions
    response = "".join(token_events)
    assert response, "Summarisation_Agent returned blank response"
    assert final_state.last_message.content
