#!/usr/bin/env python3
"""
服务器启动器
"""

from ModelServer import start_server
from ModelServer.models.utils.utils import Utils


# 使用示例
if __name__ == "__main__":        
    start_server(Utils(), port=62300)