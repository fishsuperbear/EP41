#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace hozon {
namespace netaos {
namespace adf {

using TsAlignDataType = std::shared_ptr<void>;
using TsAlignDataBundle = std::unordered_map<std::string, TsAlignDataType>;

class TsAlignImpl;

class TsAlign {
   public:
    TsAlign();
    ~TsAlign();

    using AlignSuccFunc = std::function<void(TsAlignDataBundle&)>;

    // time_window_ms: time difference among all sources should less than it
    // validity_time_ms: sources early than it will be delete
    // func:: callback on align succ
    int32_t Init(uint32_t time_window_ms, uint32_t validity_time_ms, AlignSuccFunc func);
    void Deinit();
    void RegisterSource(const std::string& source_name);
    void Push(const std::string& source_name, TsAlignDataType data, uint64_t timestamp_us);

   private:
    std::unique_ptr<TsAlignImpl> _pimpl;
};

}  // namespace adf
}  // namespace netaos
}  // namespace hozon