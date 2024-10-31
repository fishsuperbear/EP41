#pragma once

#include "adf-lite/include/base.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

class WriterImpl;
class Writer {
public:
    Writer();
    ~Writer();

    int32_t Init(const std::string& topic);
    int32_t Write(BaseDataTypePtr data);

private:
    std::unique_ptr<WriterImpl> _pimpl;
};

}
}
}
