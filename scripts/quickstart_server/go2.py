#!/usr/bin/env python3
"""
服务器启动器
"""

from ModelServer import start_server
from ModelServer.models.go2.go2 import Go2


# 使用示例
if __name__ == "__main__":        
    start_server(Go2(), port=60020)