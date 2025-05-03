import argparse
from ModelServer.server.hexmove import Hexmove_Server


parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()


hexmove_server = Hexmove_Server(is_debugging=args.debug)
