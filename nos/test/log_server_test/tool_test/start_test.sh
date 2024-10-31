#!/bin/bash

export LD_LIBRARY_PATH=/home/xiaoyu/work/netaos/nos/output/x86_2004/lib/:$LD_LIBRARY_PATH

# 执行进程 log_level_test
execute_log_level_test() {
    echo 
    echo  
    echo
    ./log_level_test
}
unset HZ_SET_LOG_LEVEL

# 1. 直接执行 log_level_test
execute_log_level_test
sleep 1

# 2. 执行命令前设置环境变量
unset HZ_SET_LOG_LEVEL
export HZ_SET_LOG_LEVEL=IGNORE.IGNORE:kOff

# 2nd execution
execute_log_level_test
sleep 1

# 3. 执行命令前设置环境变量
unset HZ_SET_LOG_LEVEL
export HZ_SET_LOG_LEVEL=IGNORE.LOG_CTX1:kCritical

# 3rd execution
execute_log_level_test
sleep 1

# 4. 执行命令前设置环境变量
unset HZ_SET_LOG_LEVEL
export HZ_SET_LOG_LEVEL=IGNORE.OP_LOG_CTX3:kCritical

# 4th execution
execute_log_level_test
sleep 1

# 5. 执行命令前设置环境变量
unset HZ_SET_LOG_LEVEL
export HZ_SET_LOG_LEVEL=APP01.IGNORE:kCritical

# 5th execution
execute_log_level_test
sleep 1

# 6. 执行命令前设置环境变量
unset HZ_SET_LOG_LEVEL
export HZ_SET_LOG_LEVEL=APP01.LOG_CTX1:kCritical

# 6th execution
execute_log_level_test
sleep 1

# 7. 执行命令后取消环境变量
unset HZ_SET_LOG_LEVEL
