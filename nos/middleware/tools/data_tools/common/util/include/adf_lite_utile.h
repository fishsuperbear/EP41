
#pragma once
#include <algorithm>
#include <iostream>
#include <map>
#include <vector>
#include "cm/include/proto_method.h"
#include "data_tools_logger.hpp"
#include "proto/test/soc/dbg_msg.pb.h"

namespace hozon {
namespace netaos {
namespace data_tool_common {

//例   /lite/lite1/info
bool IsLiteInfoTopic(const std::string& topic);
bool IsLiteMethodCMDTopic(const std::string& topic);
//例   /lite/lite1/event/workresult2
int32_t GetPartFromCmTopic(const std::string& topic, const int32_t index, std::string& value);
void RequestCommand(const std::map<std::string, std::vector<std::string>>& topics_map, const std::string& cmd, const bool status);
bool IsAdfTopic(const std::string& topic);

}  // namespace data_tool_common
}  //namespace netaos
}  //namespace hozon