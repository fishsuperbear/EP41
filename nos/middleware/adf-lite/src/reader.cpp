#include "adf-lite/include/reader.h"
#include "adf-lite/include/reader_impl.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

Reader::Reader() :
    _pimpl(new ReaderImpl()) {

}

Reader::~Reader() {

}

int32_t Reader::Init(const std::string& topic, uint32_t capcaticy) {
    return _pimpl->Init(topic, capcaticy);
}

int32_t Reader::Init(const std::string& topic, Callback cb, uint32_t capcaticy) {
    return _pimpl->Init(topic, cb, capcaticy);
}

void Reader::Pause() {
    _pimpl->Pause();
}

void Reader::Resume() {
    _pimpl->Resume();
}

BaseDataTypePtr Reader::GetLatestOneBlocking(const uint32_t timeout_ms, bool erase) {
    return _pimpl->GetLatestOneBlocking(timeout_ms, erase);
}

BaseDataTypePtr Reader::GetLatestOne(bool erase) {
    return _pimpl->GetLatestOne(erase);
}

std::vector<BaseDataTypePtr> Reader::GetLatestNdata(const size_t n, bool erase) {
    return _pimpl->GetLatestNdata(n, erase);
}

}
}
}