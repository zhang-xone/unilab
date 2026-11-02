#!/bin/bash
# 先清空所有代理变量，避免socks代理干扰DSH
unset all_proxy http_proxy https_proxy ALL_PROXY HTTP_PROXY HTTPS_PROXY

source ~/.nvm/nvm.sh
nvm use 24
# 使用已经全局安装好的dsh，不再走npx下载
/home/xxx/.nvm/versions/node/v24.22.0/bin/dsh --profile web --host 127.0.0.1 --port 3080 --no-open

