
#pragma once

#include "update_manager/common/data_def.h"

namespace hozon {
namespace netaos {
namespace update {

std::string GetDesayUpdateResultString(const int32_t& result);
std::string GetDesayUpdateStatusString(const int32_t& status);
std::string GetDesayUpdateString(const int32_t& result);
std::string GetDesayUpdateCurPartitonString(const int32_t& slot);

std::string GetTaskResultString(const uint32_t& taskResult);

}  // namespace update
}  // namespace netaos
}  // namespace hozon
