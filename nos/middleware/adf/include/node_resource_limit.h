#pragma once

#include <cstdint>
#include <string>

namespace hozon {
namespace netaos {
namespace adf {
class NodeResourceLimit {
   public:
    static int32_t LimitCpu(const std::string& group_name, uint32_t percentage);
    static int32_t LimitMem(const std::string& group_name, uint32_t memory_mb);

   private:
    static int32_t CreateDir(const std::string& dir);
    static int32_t WriteTrunc(const std::string& file, const std::string& content);
};
}  // namespace adf
}  // namespace netaos
}  // namespace hozon