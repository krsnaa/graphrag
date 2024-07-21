# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

""" 
kiku:
Based on the pyproject.toml file, when you run poetry run poe index <...args>, 
the following happens:
    index = "python -m graphrag.index" (The command is defined in the [tool.poe.tasks] section )

This command tells Poetry to run the Python module graphrag.index with any additional arguments passed.
In the Python package structure, this typically corresponds to running the __main__.py file 
inside the graphrag/index/ directory. (this file...)

So, the execution flow is:
- poetry run poe index <...args>
- Poetry executes python -m graphrag.index <...args>
- Python looks for and executes the __main__.py in the graphrag/index/ directory
- The __main__.py file parses arguments and calls index_cli() from cli.py
- The index_cli function in cli.py handles the indexing process based on the provided arguments
"""

"""The Indexing Engine package root."""

import argparse

from .cli import index_cli

# parse command-line arguments and then call the index_cli function from cli.py
if __name__ == "__main__":
    # Argparse handles parsing these arguments, converting them to the correct types,
    # and making them easily accessible in your code
    # Using argparse is generally preferred over simple input() calls for command-line programs
    # because it provides a more robust and user-friendly interface, especially for programs
    # with multiple options or complex argument structures
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="The configuration yaml file to use when running the pipeline",
        required=False,
        type=str,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Runs the pipeline with verbose logging",
        action="store_true",
    )
    parser.add_argument(
        "--memprofile",
        help="Runs the pipeline with memory profiling",
        action="store_true",
    )
    parser.add_argument(
        "--root",
        help="If no configuration is defined, the root directory to use for input data and output data. Default value: the current directory",
        # Only required if config is not defined
        required=False,
        default=".",
        type=str,
    )
    parser.add_argument(
        "--resume",
        help="Resume a given data run leveraging Parquet output files.",
        # Only required if config is not defined
        required=False,
        default=None,
        type=str,
    )
    parser.add_argument(
        "--reporter",
        help="The progress reporter to use. Valid values are 'rich', 'print', or 'none'",
        type=str,
    )
    parser.add_argument(
        "--emit",
        help="The data formats to emit, comma-separated. Valid values are 'parquet' and 'csv'. default='parquet,csv'",
        type=str,
    )
    parser.add_argument(
        "--dryrun",
        help="Run the pipeline without actually executing any steps and inspect the configuration.",
        action="store_true",
    )
    parser.add_argument("--nocache", help="Disable LLM cache.", action="store_true")
    parser.add_argument(
        "--init",
        help="Create an initial configuration in the given path.",
        action="store_true",
    )
    parser.add_argument(
        "--overlay-defaults",
        help="Overlay default configuration values on a provided configuration file (--config).",
        action="store_true",
    )

    # Namespace(config=None, verbose=False, memprofile=False, root='.', resume=None, reporter=None, emit=None, dryrun=False, nocache=False, init=False, overlay_defaults=False)
    args = parser.parse_args()

    if args.overlay_defaults and not args.config:
        parser.error("--overlay-defaults requires --config")

    index_cli(
        root=args.root,
        verbose=args.verbose or False,
        resume=args.resume,
        memprofile=args.memprofile or False,
        nocache=args.nocache or False,
        reporter=args.reporter,
        config=args.config,
        emit=args.emit,
        dryrun=args.dryrun or False,
        init=args.init or False,
        overlay_defaults=args.overlay_defaults or False,
        cli=True,
    )
