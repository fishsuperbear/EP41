#pragma once

#include <sys/time.h>
#include <string>

namespace hozon {
namespace netaos{
namespace common {
class PlatformCommonImpl {
   public:
    PlatformCommonImpl();
    ~PlatformCommonImpl();
   static int32_t Init(const std::string log_app_name, const uint32_t log_level, const uint32_t log_mode);
   static int32_t CheckTwoFrameInterval(const std::string topic_name, uint64_t& last_time);
   static int32_t GetDataTime(uint32_t &now_s, uint32_t &now_ns);
   static int32_t GetManageTime(uint32_t &now_s, uint32_t &now_ns);

    private:
};
}  // namespace common
}  // namespace hozon
}
