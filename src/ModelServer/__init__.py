from .web.client import ModelClient
from .web.server import start_server
from .config.model_server_config import MODEL_SERVER_URLS

hexmove_local = ModelClient(MODEL_SERVER_URLS['hexmove_local'])
hexmove_remote = ModelClient(MODEL_SERVER_URLS['hexmove_remote'])
hexmove = hexmove_local

utils_local = ModelClient(MODEL_SERVER_URLS['utils_local'])
utils = utils_local

go2_local = ModelClient(MODEL_SERVER_URLS['go2_local'])
go2_remote = ModelClient(MODEL_SERVER_URLS['go2_remote'])
go2 = go2_local