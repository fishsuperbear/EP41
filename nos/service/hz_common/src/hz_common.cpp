#include "hz_common.h"
#include <memory>
#include "hz_common_impl.h"

namespace hozon {
namespace netaos{
namespace common {

PlatformCommon::PlatformCommon() {}

std::unique_ptr<PlatformCommonImpl> PlatformCommon::pimpl_=std::make_unique<PlatformCommonImpl>();

PlatformCommon::~PlatformCommon() {}

int32_t PlatformCommon::Init(const std::string log_app_name, const uint32_t log_level, const uint32_t log_mode) {
    return pimpl_->Init(log_app_name, log_mode, log_level);
}

int32_t PlatformCommon::CheckTwoFrameInterval(const std::string topic_name, uint64_t& last_time) {
    return pimpl_->CheckTwoFrameInterval(topic_name, last_time);
}

int32_t PlatformCommon::GetDataTime(uint32_t& now_s, uint32_t& now_ns) {
    return pimpl_->GetDataTime(now_s, now_ns);
}

int32_t PlatformCommon::GetManageTime(uint32_t& now_s, uint32_t& now_ns) {
    return pimpl_->GetManageTime(now_s, now_ns);
}
}  // namespace common
}  // namespace hozon
}
