import pytest
from typing import List, Type
from enum import Enum
from pydantic import BaseModel, Field, create_model

from redbox.models.chain import AISettings, TaskStatus, AgentTaskBase, MultiAgentPlanBase, agent_plan_reducer

AgentEnum = Enum("AgentEnum", {"None": {"None"}})

ConfiguredAgentTask: Type[BaseModel] = create_model(
    "ConfiguredAgentTask",
    __base__=AgentTaskBase,
    agent=(
        AgentEnum,
        Field(
            description="Name of the agent to complete the task",
            default="fake_agent",
        ),
    ),
)

ConfiguredAgentPlan: Type[BaseModel] = create_model(
    "ConfiguredAgentPlan",
    __base__=MultiAgentPlanBase,
    tasks=(
        List[ConfiguredAgentTask],
        Field(
            description="A list of tasks to be carried out by agents",
            default=[ConfiguredAgentTask()],
        ),
    ),
)


def make_task(
    id: str,
    task: str = "do something",
    expected_output: str = "output",
    status: TaskStatus = TaskStatus.PENDING,
    dependencies: List[str] = None,
) -> AgentTaskBase:
    return ConfiguredAgentTask(
        id=id,
        task=task,
        expected_output=expected_output,
        status=status,
        dependencies=dependencies or [],
    )


def make_plan(*task: AgentTaskBase) -> MultiAgentPlanBase:
    return ConfiguredAgentPlan(tasks=list(task))


class TestAgentPlanReducer:
    def test_ai_settings_json_serialization_works_for_observability(self):
        settings = AISettings()

        payload = settings.model_dump(mode="json")

        assert "planner_system_prompt" not in payload
        assert settings.planner_system_prompt == settings.planner_prompt
        assert isinstance(payload["planner_question_prompt"], str)

    def test_on_none_plans(self):
        assert agent_plan_reducer(None, None) is None

    @pytest.mark.parametrize(
        "current, update",
        [
            # identical
            (
                make_plan(make_task("task1")),
                make_plan(make_task("task1")),
            ),
            # identical but reordered
            (
                make_plan(make_task("task2"), make_task("task1")),
                make_plan(make_task("task1"), make_task("task2")),
            ),
            # update empty
            (
                make_plan(make_task("task1")),
                make_plan(),
            ),
            # update none
            (
                make_plan(make_task("task1")),
                None,
            ),
            # status-only difference should not trigger replan
            (
                make_plan(make_task("task1", task="do x", status=TaskStatus.PENDING)),
                make_plan(make_task("task1", task="do x", status=TaskStatus.COMPLETED)),
            ),
        ],
    )
    def test_returns_current(self, current, update):
        result = agent_plan_reducer(current, update)
        assert result == current

    @pytest.mark.parametrize(
        "current, update",
        [
            # different task count
            (
                make_plan(make_task("task1")),
                make_plan(make_task("task1"), make_task("task2")),
            ),
            (
                make_plan(make_task("task1"), make_task("task2")),
                make_plan(make_task("task1")),
            ),
            # different ids
            (
                make_plan(make_task("task1")),
                make_plan(make_task("task2")),
            ),
            # all new ids
            (
                make_plan(make_task("task1"), make_task("task2")),
                make_plan(make_task("task3"), make_task("task4")),
            ),
            # different task content
            (
                make_plan(make_task("task1", task="summarise doc")),
                make_plan(make_task("task1", task="search the web")),
            ),
            # different expected output
            (
                make_plan(make_task("task1", expected_output="summary")),
                make_plan(make_task("task1", expected_output="table")),
            ),
            # different dependencies
            (
                make_plan(
                    make_task("task1"),
                    make_task("task2", dependencies=[]),
                ),
                make_plan(
                    make_task("task1"),
                    make_task("task2", dependencies=["task1"]),
                ),
            ),
            # current none
            (
                None,
                make_plan(make_task("task1")),
            ),
        ],
    )
    def test_returns_update(self, current, update):
        result = agent_plan_reducer(current, update)
        assert result == update

    @pytest.mark.parametrize(
        "current_status, update_status, expected_status",
        [
            (
                TaskStatus.PENDING,
                TaskStatus.RUNNING,
                TaskStatus.RUNNING,
            ),
            (
                TaskStatus.RUNNING,
                TaskStatus.COMPLETED,
                TaskStatus.COMPLETED,
            ),
            (
                TaskStatus.COMPLETED,
                TaskStatus.PENDING,
                TaskStatus.COMPLETED,
            ),
        ],
    )
    def test_status_progression(
        self,
        current_status,
        update_status,
        expected_status,
    ):
        current = make_plan(make_task("task1", status=current_status))
        update = make_plan(make_task("task1", status=update_status))

        result = agent_plan_reducer(current, update)

        assert result.tasks[0].status == expected_status

    @pytest.mark.parametrize(
        "current, update, expected_statuses",
        [
            (
                make_plan(
                    make_task("task1", status=TaskStatus.PENDING),
                    make_task("task2", status=TaskStatus.PENDING),
                ),
                make_plan(
                    make_task("task1", status=TaskStatus.COMPLETED),
                    make_task("task2", status=TaskStatus.PENDING),
                ),
                [TaskStatus.COMPLETED, TaskStatus.PENDING],
            ),
            (
                make_plan(
                    make_task("task1", status=TaskStatus.PENDING),
                    make_task("task2", status=TaskStatus.PENDING),
                ),
                make_plan(
                    make_task("task1", status=TaskStatus.RUNNING),
                    make_task("task2", status=TaskStatus.COMPLETED),
                ),
                [TaskStatus.RUNNING, TaskStatus.COMPLETED],
            ),
        ],
    )
    def test_multiple_status_updates(
        self,
        current,
        update,
        expected_statuses,
    ):
        result = agent_plan_reducer(current, update)

        actual = [task.status for task in result.tasks]
        assert actual == expected_statuses

    def test_status_updates_mutate_current(self):
        current = make_plan(make_task("task1", status=TaskStatus.PENDING))
        update = make_plan(make_task("task1", status=TaskStatus.COMPLETED))

        result = agent_plan_reducer(current, update)

        assert result is current
        assert result.tasks[0].status == TaskStatus.COMPLETED

    def test_replan_preserves_updated_content(self):
        current = make_plan(make_task("task1", task="old"))
        update = make_plan(make_task("task1", task="new"))

        result = agent_plan_reducer(current, update)

        assert result is update
        assert result.tasks[0].task == "new"

    def test_large_plan_partial_status_update(self):
        current_tasks = [make_task(f"task{i}", task=f"task {i}") for i in range(10)]

        update_tasks = [
            make_task(
                f"task{i}",
                task=f"task {i}",
                status=(TaskStatus.COMPLETED if i < 5 else TaskStatus.PENDING),
            )
            for i in range(10)
        ]

        result = agent_plan_reducer(
            make_plan(*current_tasks),
            make_plan(*update_tasks),
        )

        for i, task in enumerate(result.tasks):
            expected = TaskStatus.COMPLETED if i < 5 else TaskStatus.PENDING
            assert task.status == expected
