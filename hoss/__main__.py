""" Run the Harmony OPeNDAP SubSetter Adapter via the Harmony CLI. """
from argparse import ArgumentParser
from sys import argv

from harmony import is_harmony_cli, run_cli, setup_cli

from hoss.adapter import HossAdapter


def main(arguments: list[str]):
    """ Parse command line arguments and invoke the appropriate method to
        respond to them

    """
    parser = ArgumentParser(prog='harmony-opendap-subsetter',
                            description='Run Harmony OPeNDAP SubSetter.')

    setup_cli(parser)
    harmony_arguments, _ = parser.parse_known_args(arguments[1:])

    if is_harmony_cli(harmony_arguments):
        run_cli(parser, harmony_arguments, HossAdapter)
    else:
        parser.error('Only --harmony CLIs are supported')


if __name__ == '__main__':
    main(argv)
