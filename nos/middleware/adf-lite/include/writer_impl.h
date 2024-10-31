#pragma once

#include "adf-lite/include/topology.h"
#include "adf-lite/include/writer.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

class WriterImpl {
public:
    WriterImpl(Writer* writer);
    ~WriterImpl();

    int32_t Init(const std::string& topic);
    int32_t Write(BaseDataTypePtr data);

private:
    Writer* _writer;
    std::string _topic;
    void ParseProtoHeader(BaseDataTypePtr& data);
};

}
}
}