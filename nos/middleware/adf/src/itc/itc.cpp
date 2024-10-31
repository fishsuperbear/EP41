#include "adf/include/itc/itc.h"
#include "adf/include/itc/itc_impl.h"

namespace hozon {
namespace netaos {
namespace adf {

ITCWriter::ITCWriter() : _pimpl(new ITCWriterImpl) {}

ITCWriter::~ITCWriter() {}

int32_t ITCWriter::Init(const std::string& topic_name) {
    return _pimpl->Init(topic_name);
}

void ITCWriter::Deinit() {
    _pimpl->Deinit();
}

void ITCWriter::Write(ITCDataType data) {
    _pimpl->Write(data);
}

ITCReader::ITCReader() : _pimpl(new ITCReaderImpl) {}

ITCReader::~ITCReader() {}

int32_t ITCReader::Init(const std::string& topic_name, uint32_t capacity) {
    return _pimpl->Init(topic_name, capacity);
}

int32_t ITCReader::Init(const std::string& topic_name, CallbackFunc callback, uint32_t capacity) {
    return _pimpl->Init(topic_name, callback, capacity);
}

void ITCReader::Deinit() {
    _pimpl->Deinit();
}

ITCDataType ITCReader::Take() {
    return _pimpl->Take();
}

ITCDataType ITCReader::Take(const uint32_t timeout_ms) {
    return _pimpl->Take(timeout_ms);
}

}  // namespace adf
}  // namespace netaos
}  // namespace hozon