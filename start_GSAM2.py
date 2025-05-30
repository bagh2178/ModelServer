import argparse
from ModelServer.server.GSAM2 import GSAM2_Server

parser = argparse.ArgumentParser()
parser.add_argument('--server_ip', default='127.0.0.1')
parser.add_argument('--server_port', default=7102)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()


GSAM2_server = GSAM2_Server(server_ip=args.server_ip, server_port=int(args.server_port), is_debugging=args.debug)
