#!/bin/bash

ps aux | grep -E '/opt/platform/mdc_platform|/usr/bin/mdc/base-plat/log_daemon/bin/log_daemon' \
| grep -v grep | awk '{print $2}' | xargs kill -9
echo "Stop finish!"