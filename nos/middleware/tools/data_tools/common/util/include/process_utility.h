/*
 * Copyright (c) hozonauto. 2021-2022. All rights reserved.
 * Description: CostSpan definition
 */

#pragma once

#include <string>

namespace hozon {
namespace netaos {
namespace data_tool_common {

class ProcessUtility {
public:

    static void SetThreadName(std::string name);
    static bool IsBackground();
};

}
}
}