
import argparse
import json 
from .arguments import get_args_parser

def get_json_args(config_path=None):
    conf_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)
    conf_parser.add_argument("-c", "--conf_file",  default=None, help="Specify config file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()
    defaults = {}
    if config_path:
        with open(config_path, 'r') as f:
            defaults = json.load(f)
    if args.conf_file:
        with open(args.conf_file, 'r') as f:
            defaults = json.load(f)
    # parser = argparse.ArgumentParser(parents=[conf_parser])
    parser = argparse.ArgumentParser('GFNet training and evaluation script', parents=[get_args_parser()])
    parser.set_defaults(**defaults)

    config = parser.parse_known_args(remaining_argv)[0]
    config.config_file = args.conf_file
    return config