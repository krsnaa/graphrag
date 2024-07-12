# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing load_workflows, create_workflow, _get_steps_for_workflow and _remove_disabled_steps methods definition."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from datashaper import Workflow

from graphrag.index.errors import (
    NoWorkflowsDefinedError,
    UndefinedWorkflowError,
    UnknownWorkflowError,
)
from graphrag.index.utils import topological_sort

from .default_workflows import default_workflows as _default_workflows
from .typing import VerbDefinitions, WorkflowDefinitions, WorkflowToRun

if TYPE_CHECKING:
    from graphrag.index.config import (
        PipelineWorkflowConfig,
        PipelineWorkflowReference,
        PipelineWorkflowStep,
    )

anonymous_workflow_count = 0

VerbFn = Callable[..., Any]
log = logging.getLogger(__name__)


class LoadWorkflowResult(NamedTuple):
    """A workflow loading result object."""

    workflows: list[WorkflowToRun]
    """The loaded workflow names in the order they should be run."""

    dependencies: dict[str, list[str]]
    """A dictionary of workflow name to workflow dependencies."""


def load_workflows(
    workflows_to_load: list[PipelineWorkflowReference],
    additional_verbs: VerbDefinitions | None = None,
    additional_workflows: WorkflowDefinitions | None = None,
    memory_profile: bool = False,
) -> LoadWorkflowResult:
    """
    Load the given workflows.

    Load and prepare workflows for execution in the pipeline.

    This function processes a list of workflow references, resolves their dependencies,
    and prepares them for execution. It can also incorporate additional custom verbs
    and workflows.

    Args:
        - workflows_to_load - The workflows to load
        - additional_verbs - The list of custom verbs available to the workflows
        - additional_workflows - The list of custom workflows
    Returns:
        - output[0] - The loaded workflow names in the order they should be run
        - output[1] - A dictionary of workflow name to workflow dependencies

    Notes:
    ------
    - This function handles both named and anonymous workflows.
    - It resolves dependencies between workflows, ensuring they are executed in the correct order.
    - If a workflow depends on another that isn't explicitly included, it will be automatically added.
    - The function uses topological sorting to determine the final execution order.
    - Memory profiling can be enabled for performance analysis.
    """
    # Initialize the workflow graph - a dictionary of workflow names to WorkflowToRun objects
    workflow_graph: dict[str, WorkflowToRun] = {}

    global anonymous_workflow_count

    for reference in workflows_to_load:
        name = reference.name
        is_anonymous = name is None or name.strip() == ""
        if is_anonymous:
            name = f"Anonymous Workflow {anonymous_workflow_count}"
            anonymous_workflow_count += 1

        # Python's type hinting system - It tells type checking tools that
        # name should be treated as a string from this point forward in the code.
        name = cast(str, name)

        config = reference.config
        workflow = create_workflow(
            name or "MISSING NAME!",
            reference.steps,
            config,
            additional_verbs,
            additional_workflows,
        )
        workflow_graph[name] = WorkflowToRun(workflow, config=config or {})

    # Backfill any missing workflows (dependencies) into the workflow graph
    for name in list(workflow_graph.keys()):
        workflow = workflow_graph[name]

        # create a new list of workflow.workflow.dependencies and remove the "workflow:" prefix
        deps = [
            d.replace("workflow:", "")
            for d in workflow.workflow.dependencies
            if d.startswith("workflow:")
        ]

        # Add any non- explicitly included workflows that are a dependency to existing workflows
        for dependency in deps:
            if dependency not in workflow_graph:
                #  create a new dictionary that's a combination of the dependency and the workflow.config
                reference = {"name": dependency, **workflow.config}
                workflow_graph[dependency] = WorkflowToRun(
                    workflow=create_workflow(
                        dependency,
                        config=reference,
                        additional_verbs=additional_verbs,
                        additional_workflows=additional_workflows,
                        memory_profile=memory_profile,
                    ),
                    config=reference,
                )

    # Run workflows in order of dependencies
    def filter_wf_dependencies(name: str) -> list[str]:
        # list comprehension - create a list of external dependencies for a given workflow
        externals = [
            e.replace("workflow:", "")
            for e in workflow_graph[name].workflow.dependencies
        ]
        return [e for e in externals if e in workflow_graph]

    """ 
    {
        'workflow1': ['dep1', 'dep2'],
        'workflow2': ['dep3'],
        'workflow3': ['dep1', 'dep4'],
        # ... and so on for all workflows in workflow_graph
    }
    """
    # dictionary comprehension - get a list of valid workflow dependencies for a given workflow
    task_graph = {name: filter_wf_dependencies(name) for name in workflow_graph}

    # sort the workflows in the order they should be run, based on their dependencies
    workflow_run_order = topological_sort(task_graph)

    workflows = [workflow_graph[name] for name in workflow_run_order]
    log.info("Workflow Run Order: %s", workflow_run_order)

    return LoadWorkflowResult(workflows=workflows, dependencies=task_graph)


def create_workflow(
    name: str,
    steps: list[PipelineWorkflowStep] | None = None,
    config: PipelineWorkflowConfig | None = None,
    additional_verbs: VerbDefinitions | None = None,
    additional_workflows: WorkflowDefinitions | None = None,
    memory_profile: bool = False,
) -> Workflow:
    """Create a workflow from the given config."""
    additional_workflows = {
        **_default_workflows,
        **(additional_workflows or {}),
    }
    steps = steps or _get_steps_for_workflow(name, config, additional_workflows)
    steps = _remove_disabled_steps(steps)
    return Workflow(
        verbs=additional_verbs or {},
        schema={
            "name": name,
            "steps": steps,
        },
        validate=False,
        memory_profile=memory_profile,
    )


def _get_steps_for_workflow(
    name: str | None,
    config: PipelineWorkflowConfig | None,
    workflows: dict[str, Callable] | None,
) -> list[PipelineWorkflowStep]:
    """Get the steps for the given workflow config."""
    if config is not None and "steps" in config:
        return config["steps"]

    if workflows is None:
        raise NoWorkflowsDefinedError

    if name is None:
        raise UndefinedWorkflowError

    if name not in workflows:
        raise UnknownWorkflowError(name)

    return workflows[name](config or {})


def _remove_disabled_steps(
    steps: list[PipelineWorkflowStep],
) -> list[PipelineWorkflowStep]:
    return [step for step in steps if step.get("enabled", True)]
