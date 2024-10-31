#include "adf/include/ts_align/ts_align.h"
#include "adf/include/ts_align/ts_align_impl.h"

namespace hozon {
namespace netaos {
namespace adf {

TsAlign::TsAlign() : _pimpl(new TsAlignImpl) {}

TsAlign::~TsAlign() {}

int32_t TsAlign::Init(uint32_t time_window_ms, uint32_t validity_time_ms, AlignSuccFunc func) {
    return _pimpl->Init(time_window_ms, validity_time_ms, func);
}

void TsAlign::Deinit() {
    _pimpl->Deinit();
}

void TsAlign::RegisterSource(const std::string& source_name) {
    _pimpl->RegisterSource(source_name);
}

void TsAlign::Push(const std::string& source_name, TsAlignDataType data, uint64_t timestamp_us) {
    _pimpl->Push(source_name, data, timestamp_us);
}

}  // namespace adf
}  // namespace netaos
}  // namespace hozon