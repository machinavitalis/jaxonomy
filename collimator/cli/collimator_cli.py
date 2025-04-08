# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

import click

from collimator.cli.cli_run import collimator_run
from collimator.cli.run_optimization import collimator_optimize


@click.group()
def cli():
    pass


cli.add_command(collimator_run)
cli.add_command(collimator_optimize)

if __name__ == "__main__":
    cli()
