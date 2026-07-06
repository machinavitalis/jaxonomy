# SPDX-License-Identifier: MIT

import click

from jaxonomy.cli.cli_run import jaxonomy_run
from jaxonomy.cli.run_optimization import jaxonomy_optimize
from jaxonomy.cli.run_variants import jaxonomy_variants


@click.group()
def cli():
    pass


cli.add_command(jaxonomy_run)
cli.add_command(jaxonomy_optimize)
cli.add_command(jaxonomy_variants)

if __name__ == "__main__":
    cli()
