#pragma once

#include "adf-lite/include/base.h"

namespace hozon {
namespace netaos {
namespace adf_lite {
    
class ReaderImpl;

class Reader {
public:
    using Callback = std::function<void(BaseDataTypePtr data)>;

    Reader();
    ~Reader();

    int32_t Init(const std::string& topic, uint32_t capcaticy);
    int32_t Init(const std::string& topic, Callback cb, uint32_t capcaticy);

    void Pause();
    void Resume();

    // DO NOT call functions below in callback
    BaseDataTypePtr GetLatestOneBlocking(const uint32_t timeout_ms, bool erase);
    BaseDataTypePtr GetLatestOne(bool erase);
    std::vector<BaseDataTypePtr> GetLatestNdata(const size_t n, bool erase);

private:
    std::unique_ptr<ReaderImpl> _pimpl;
};

}
}
}