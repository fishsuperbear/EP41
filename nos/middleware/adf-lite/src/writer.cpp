#include "adf-lite/include/writer.h"
#include "adf-lite/include/writer_impl.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

Writer::Writer() :
    _pimpl(new WriterImpl(this)) {

}

Writer::~Writer() {

}

int32_t Writer::Init(const std::string& topic) {
    return _pimpl->Init(topic);
}

int32_t Writer::Write(BaseDataTypePtr data) {
    return _pimpl->Write(data);
}

}
}
}