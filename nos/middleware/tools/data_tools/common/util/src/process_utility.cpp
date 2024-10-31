#include "process_utility.h"

#include <iostream>
#include <csignal>
#include <unistd.h>

namespace hozon {
namespace netaos {
namespace data_tool_common {

void ProcessUtility::SetThreadName(std::string name) {

    if (name.size() > 16) {
        name = name.substr(name.size() - 15, 15);
    }
    pthread_setname_np(pthread_self(), name.c_str());
}

bool ProcessUtility::IsBackground() {
    pid_t pid = getpid(); // 获取当前进程的PID

    // 获取当前进程组的ID
    pid_t pgid = getpgrp();

    // 获取前台进程组的ID
    pid_t tc_pgid = tcgetpgrp(STDIN_FILENO);

    if (pgid == tc_pgid) {
        return false;
    }

    return true;
}

}
}
}