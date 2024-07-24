# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Main definition."""

import asyncio
import json
import logging
import platform
import sys
import time
import warnings
from pathlib import Path

from graphrag.config import (
    create_graphrag_config,
)
from graphrag.index import PipelineConfig, create_pipeline_config
from graphrag.index.cache import NoopPipelineCache
from graphrag.index.progress import (
    NullProgressReporter,
    PrintProgressReporter,
    ProgressReporter,
)
from graphrag.index.progress.rich import RichProgressReporter
from graphrag.index.run import run_pipeline_with_config

from .emit import TableEmitterType
from .graph.extractors.claims.prompts import CLAIM_EXTRACTION_PROMPT
from .graph.extractors.community_reports.prompts import COMMUNITY_REPORT_PROMPT
from .graph.extractors.graph.prompts import GRAPH_EXTRACTION_PROMPT
from .graph.extractors.summarize.prompts import SUMMARIZE_PROMPT
from .init_content import INIT_DOTENV, INIT_YAML

# Ignore warnings from numba
warnings.filterwarnings("ignore", message=".*NumbaDeprecationWarning.*")

log = logging.getLogger(__name__)


def redact(input: dict) -> str:
    """Sanitize the config json."""

    # Redact any sensitive configuration
    def redact_dict(input: dict) -> dict:
        if not isinstance(input, dict):
            return input

        result = {}
        for key, value in input.items():
            if key in {
                "api_key",
                "connection_string",
                "container_name",
                "organization",
            }:
                if value is not None:
                    result[key] = f"REDACTED, length {len(value)}"
            elif isinstance(value, dict):
                result[key] = redact_dict(value)
            elif isinstance(value, list):
                result[key] = [redact_dict(i) for i in value]
            else:
                result[key] = value
        return result

    redacted_dict = redact_dict(input)
    return json.dumps(redacted_dict, indent=4)


def index_cli(
    root: str,
    init: bool,
    verbose: bool,
    resume: str | None,
    memprofile: bool,
    nocache: bool,
    reporter: str | None,
    config: str | None,
    emit: str | None,
    dryrun: bool,
    overlay_defaults: bool,
    cli: bool = False,
):
    """
    Run the pipeline with the given config.
    kiku:
    Execute the main indexing pipeline with the specified configuration and options.

    This function serves as the entry point for the indexing CLI, setting up the
    environment, loading configurations, and running the indexing process.

    Parameters:
    -----------
    root : str
        The root directory for input data and output data.
    init : bool
        If True, initialize a new project in the specified root directory.
    verbose : bool
        If True, enable verbose logging.
    resume : str | None
        If provided, resume a previous data run using the specified run ID.
    memprofile : bool
        If True, enable memory profiling during execution.
    nocache : bool
        If True, disable the LLM cache.
    reporter : str | None
        The type of progress reporter to use. Options: 'rich', 'print', or 'none'.
    config : str | None
        Path to a custom configuration file to use.
    emit : str | None
        Comma-separated list of data formats to emit (e.g., 'parquet,csv').
    dryrun : bool
        If True, perform a dry run without executing any steps, useful for config inspection.
    overlay_defaults : bool
        If True, overlay default configuration values on the provided config file.
    cli : bool, optional
        If True, indicate that the function is being run from the CLI (default: False).

    Returns:
    --------
    None

    Notes:
    ------
    - Configuration and Setup (in index_cli):
        A run ID is generated or retrieved (if resuming).
        The function can handle both new runs and resuming previous runs.
        Logging is set up based on the run ID and verbosity settings.
        A progress reporter is created based on the specified type.
        If --init was specified, project initialization occurs and the program exits. (_initialize_project_at)
    - Pipeline Configuration:
        The pipeline configuration is loaded or created using create_pipeline_config() - (inside create_pipeline_config.py).
        This involves reading from config files or environment variables and setting up various components (input, storage, cache, etc.).
        Also, the workflows are defined and added to the configuration
        workflows=[
            *_document_workflows(settings, embedded_fields),
            *_text_unit_workflows(settings, covariates_enabled, embedded_fields),
            *_graph_workflows(settings, embedded_fields),
            *_community_workflows(settings, covariates_enabled, embedded_fields),
            *(_covariate_workflows(settings) if covariates_enabled else []),
        ]. The workflow names are defined in workflows/default_workflows.py, each of which returns a list of steps.
        These steps are individual operations (often called "verbs" in data processing pipelines) that are executed in order.
        Here is the sequence of steps after topological sorting:
            1- create_base_text_units
            2- create_base_extracted_entities
            3- create_final_covariates (if enabled)
            4- create_summarized_entities
            5- join_text_units_to_covariate_ids (if enabled)
            6- create_base_entity_graph
            7- create_final_entities
            8- create_final_nodes
            9- create_final_communities
            10- join_text_units_to_entity_ids
            11- create_final_relationships
            12- join_text_units_to_relationship_ids
            13- create_final_community_reports
            14- create_final_text_units
            15- create_base_documents
            16- create_final_documents
    - Async Workflow Execution:
        The _run_workflow_async() function is setup and called to execute the pipeline asynchronously.
        Signal handlers are set up for graceful termination.
        The appropriate event loop is configured (uvloop for non-Windows, asyncio for Windows).
    - Pipeline Execution:
        graphrag/index/run.py::run_pipeline_with_config() is called within the async context in execute().
        That in turn, calls run_pipeline(), which in turn, calls load_workflows() in load.py, which
            determines the execution order of the workflows using topological_sort to account for dependencies.
        This function iterates through the indexing pipeline workflows, executing each in order.
    - Progress Reporting and Error Handling:
        Throughout the execution, progress is reported using the configured reporter.
        Errors are caught, logged, and may modify the encountered_errors flag.
    - Workflow Completion:
        As each workflow completes, its results are yielded back to the main execution loop.
    - Cleanup and Finalization:
        After all workflows are complete, or if interrupted, cleanup processes run.
        Final status messages are reported (success or errors encountered).
    - Program Termination:
        If running from CLI (cli=True), sys.exit is called with an appropriate exit code.
        Otherwise, the function returns, allowing for potential further processing if called programmatically.
    - Various options allow for customization of the indexing process, including
      output formats, caching, and reporting.
    """

    # 20240705-030018
    run_id = resume or time.strftime("%Y%m%d-%H%M%S")
    _enable_logging(root, run_id, verbose)
    progress_reporter = _get_progress_reporter(reporter)

    # This will only create two files: .env and settings.yaml
    # in the --root directory and terminate the program
    if init:
        _initialize_project_at(root, progress_reporter)
        sys.exit(0)

    # If overlay_defaults is True, the default configuration values
    # will be overlaid on the provided configuration file
    # otherwise, the provided configuration values will be used

    # in either case, we get back a fully populated PipelineConfig object
    # ready for use in the pipeline.
    if overlay_defaults:
        pipeline_config: str | PipelineConfig = _create_default_config(
            root, config, verbose, dryrun or False, progress_reporter
        )
    else:
        pipeline_config: str | PipelineConfig = config or _create_default_config(
            root, None, verbose, dryrun or False, progress_reporter
        )

    cache = NoopPipelineCache() if nocache else None
    pipeline_emit = emit.split(",") if emit else None
    encountered_errors = False

    def _run_workflow_async() -> None:
        """
        Execute the pipeline workflow asynchronously.

        This function sets up and runs the main pipeline workflow in an asynchronous manner.
        It handles signal interrupts, manages the event loop, and executes the pipeline
        using the configured settings.

        Key Features:
        - Sets up signal handlers for graceful termination (SIGINT, SIGHUP)
        - Configures and uses the appropriate event loop based on the operating system
        - Executes the pipeline workflow using run_pipeline_with_config
        - Handles progress reporting and error logging during execution

        Global Effects:
        - Modifies the global 'encountered_errors' flag based on workflow execution results

        Notes:
        - Uses uvloop on non-Windows systems for improved performance
        - Falls back to asyncio's default loop on Windows systems
        - Utilizes nest_asyncio on Windows for compatibility in certain environments

        Raises:
        - May raise various exceptions related to pipeline execution, which are caught
          and logged by the error handling mechanism

        This function is typically called within the main execution flow of the indexing process
        and shouldn't be called directly under normal circumstances.
        """
        import signal

        # graceful termination signal handler
        def handle_signal(signum, _):
            # Handle the signal here
            progress_reporter.info(f"Received signal {signum}, exiting...")
            progress_reporter.dispose()
            for task in asyncio.all_tasks():
                task.cancel()
            progress_reporter.info("All tasks cancelled. Exiting...")

        # Register signal handlers for SIGINT and SIGHUP
        signal.signal(signal.SIGINT, handle_signal)

        if sys.platform != "win32":
            signal.signal(signal.SIGHUP, handle_signal)

        async def execute():
            """
            Asynchronously execute the pipeline workflows.

            This function is the core of the asynchronous pipeline execution. It iterates
            through all configured workflows, running each one and handling their outputs
            and potential errors.

            Global Effects:
            - Modifies the global 'encountered_errors' flag if any errors occur during execution.

            Workflow Execution:
            - Iterates through workflows defined in the pipeline configuration.
            - Runs each workflow using run_pipeline_with_config().
            - Captures and processes the output of each workflow.

            Error Handling:
            - Logs errors encountered during workflow execution.
            - Sets the global 'encountered_errors' flag if any errors occur.

            Progress Reporting:
            - Uses the configured progress reporter to update on workflow status.
            - Reports successful completion or error for each workflow.

            Notes:
            - This function is designed to be run within an asynchronous context.
            - It's typically called by _run_workflow_async and shouldn't be invoked directly.

            Returns:
            -------
            None

            Raises:
            ------
            No exceptions are raised directly by this function, as all errors are caught
            and handled internally. However, it may propagate exceptions from the underlying
            run_pipeline_with_config function if they are not handled there.
            """
            nonlocal encountered_errors
            # iterates through the configured workflows, executing each in order.
            async for output in run_pipeline_with_config(
                pipeline_config,
                run_id=run_id,
                memory_profile=memprofile,
                cache=cache,
                # Throughout the execution, progress is reported using the configured reporter.
                progress_reporter=progress_reporter,
                # JSON, CSV, or Parquet
                emit=(
                    [TableEmitterType(e) for e in pipeline_emit]
                    if pipeline_emit
                    else None
                ),
                is_resume_run=bool(resume),
            ):
                # Errors are caught, logged, and modify the encountered_errors flag.
                if output.errors and len(output.errors) > 0:
                    encountered_errors = True
                    progress_reporter.error(output.workflow)
                else:
                    progress_reporter.success(output.workflow)

                progress_reporter.info(str(output.result))

        # The appropriate event loop is configured (uvloop for non-Windows, asyncio for Windows).
        if platform.system() == "Windows":
            import nest_asyncio  # type: ignore Ignoring because out of windows this will cause an error

            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            loop.run_until_complete(execute())
        elif sys.version_info >= (3, 11):
            import uvloop  # type: ignore Ignoring because on windows this will cause an error

            with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:  # type: ignore Ignoring because minor versions this will throw an error
                runner.run(execute())
        else:
            # platform is not windows and python version is less than 3.11
            # this block might seem unreachable because of the python version setup in VSCode
            import uvloop  # type: ignore Ignoring because on windows this will cause an error

            uvloop.install()
            asyncio.run(execute())

    # set up and execute the pipeline asynchronously
    _run_workflow_async()

    progress_reporter.stop()

    # Final status messages are reported (success or errors encountered).
    if encountered_errors:
        progress_reporter.error(
            "Errors occurred during the pipeline run, see logs for more details."
        )
    else:
        progress_reporter.success("All workflows completed successfully.")

    # If running from CLI (cli=True), sys.exit is called with an appropriate exit code.
    if cli:
        sys.exit(1 if encountered_errors else 0)


def _initialize_project_at(path: str, reporter: ProgressReporter) -> None:
    """Initialize the project at the given path."""
    reporter.info(f"Initializing project at {path}")
    root = Path(path)
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    settings_yaml = root / "settings.yaml"
    if settings_yaml.exists():
        msg = f"Project already initialized at {root}"
        raise ValueError(msg)

    dotenv = root / ".env"
    if not dotenv.exists():
        with settings_yaml.open("w") as file:
            file.write(INIT_YAML)

    with dotenv.open("w") as file:
        file.write(INIT_DOTENV)

    prompts_dir = root / "prompts"
    if not prompts_dir.exists():
        prompts_dir.mkdir(parents=True, exist_ok=True)

    entity_extraction = prompts_dir / "entity_extraction.txt"
    if not entity_extraction.exists():
        with entity_extraction.open("w") as file:
            file.write(GRAPH_EXTRACTION_PROMPT)

    summarize_descriptions = prompts_dir / "summarize_descriptions.txt"
    if not summarize_descriptions.exists():
        with summarize_descriptions.open("w") as file:
            file.write(SUMMARIZE_PROMPT)

    claim_extraction = prompts_dir / "claim_extraction.txt"
    if not claim_extraction.exists():
        with claim_extraction.open("w") as file:
            file.write(CLAIM_EXTRACTION_PROMPT)

    community_report = prompts_dir / "community_report.txt"
    if not community_report.exists():
        with community_report.open("w") as file:
            file.write(COMMUNITY_REPORT_PROMPT)


def _create_default_config(
    root: str,
    config: str | None,
    verbose: bool,
    dryrun: bool,
    reporter: ProgressReporter,
) -> PipelineConfig:
    """
    Overlay default values on an existing config or create a default config if none is provided.

    Calls:
    ------
    - _read_config_parameters: Read and parse configuration parameters from YAML, JSON, or environment variables.
    - create_pipeline_config: Create a PipelineConfig object from the configuration parameters.
        - the workflows are defined and added to the configuration here

    Notes:
    ------
    - If a config file is specified but doesn't exist, it raises an error.
    - The function logs the configuration details if verbose mode is enabled.
    - In case of a dry run, it logs the configuration and exits without further execution.
    """
    if config and not Path(config).exists():
        msg = f"Configuration file {config} does not exist"
        raise ValueError

    if not Path(root).exists():
        msg = f"Root directory {root} does not exist"
        raise ValueError(msg)

    parameters = _read_config_parameters(root, config, reporter)
    log.info(
        "using default configuration: %s",
        redact(parameters.model_dump()),
    )

    if verbose or dryrun:
        reporter.info(f"Using default configuration: {redact(parameters.model_dump())}")

    # we get back a fully populated PipelineConfig object ready for use in the pipeline.
    result = create_pipeline_config(parameters, verbose)

    if verbose or dryrun:
        reporter.info(f"Final Config: {redact(result.model_dump())}")

    if dryrun:
        reporter.info("dry run complete, exiting...")
        sys.exit(0)

    return result


def _read_config_parameters(root: str, config: str | None, reporter: ProgressReporter):
    """
    Read and parse configuration parameters from YAML, JSON, or environment variables.

    This function attempts to load configuration settings in the following order:
    1. From a specified YAML file (settings.yaml or settings.yml)
    2. From a specified JSON file (settings.json)
    3. From environment variables if no file is found
    """
    # create a Path object from the root parameter
    _root = Path(root)

    # if config is provided and if it is a yaml or yml file
    settings_yaml = (
        Path(config)
        if config and Path(config).suffix in [".yaml", ".yml"]
        else _root / "settings.yaml"
    )

    # if settings.yaml does not exist, seti it to settings.yml
    if not settings_yaml.exists():
        settings_yaml = _root / "settings.yml"

    settings_json = (
        Path(config)
        if config and Path(config).suffix == ".json"
        else _root / "settings.json"
    )

    if settings_yaml.exists():
        reporter.success(f"Reading settings from {settings_yaml}")
        with settings_yaml.open("r") as file:
            import yaml

            data = yaml.safe_load(file)
            return create_graphrag_config(data, root)

    if settings_json.exists():
        reporter.success(f"Reading settings from {settings_json}")
        with settings_json.open("r") as file:
            import json

            data = json.loads(file.read())
            return create_graphrag_config(data, root)

    reporter.success("Reading settings from environment variables")
    return create_graphrag_config(root_dir=root)


def _get_progress_reporter(reporter_type: str | None) -> ProgressReporter:
    if reporter_type is None or reporter_type == "rich":
        return RichProgressReporter("GraphRAG Indexer ")
    if reporter_type == "print":
        return PrintProgressReporter("GraphRAG Indexer ")
    if reporter_type == "none":
        return NullProgressReporter()

    msg = f"Invalid progress reporter type: {reporter_type}"
    raise ValueError(msg)


def _enable_logging(root_dir: str, run_id: str, verbose: bool) -> None:
    logging_file = (
        Path(root_dir) / "output" / run_id / "reports" / "indexing-engine.log"
    )
    logging_file.parent.mkdir(parents=True, exist_ok=True)

    logging_file.touch(exist_ok=True)

    logging.basicConfig(
        filename=str(logging_file),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if verbose else logging.INFO,
    )
