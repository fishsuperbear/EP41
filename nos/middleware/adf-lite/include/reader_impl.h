#pragma once

#include <thread>

#include "adf-lite/include/reader.h"
#include "adf-lite/include/cv_queue.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

class ReaderImpl {
public:
    ReaderImpl();
    ~ReaderImpl();

    int32_t Init(const std::string& topic, uint32_t capcaticy);
    int32_t Init(const std::string& topic, Reader::Callback cb, uint32_t capcaticy);

    void Pause();
    void Resume();

    // DO NOT call functions below in callback
    BaseDataTypePtr GetLatestOneBlocking(const uint32_t timeout_ms, bool erase);
    BaseDataTypePtr GetLatestOne(bool erase = true);
    std::vector<BaseDataTypePtr> GetLatestNdata(const size_t n, bool erase);

private:
    std::string _topic;
    Reader::Callback _callback;
    std::shared_ptr<CVSizeLimitQueue<BaseDataTypePtr>> _queue;
    std::shared_ptr<std::thread> _recv_thread;
    bool _need_stop = false;
};

}
}
}