/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 */

#ifndef SRC_PLOG_API_PLOG_CMD_PLOGCMD_H
#define SRC_PLOG_API_PLOG_CMD_PLOGCMD_H

#include <string>
#include <vector>

namespace rbs {
namespace plog {
class PlogCmd {
public:
    static int MainEntrance(const std::vector<std::string>& cmdList);
};
}
}

#endif // SRC_PLOG_API_PLOG_CMD_PLOGCMD_H
