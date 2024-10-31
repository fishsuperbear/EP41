#include "adf-lite/include/reader_impl.h"
#include "adf-lite/include/topology.h"
#include "adf-lite/include/adf_lite_internal_logger.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

ReaderImpl::ReaderImpl() {

}

ReaderImpl::~ReaderImpl() {
    _need_stop = true;
    if (_recv_thread) {
        _recv_thread->join();
    }
    
    _queue->Exit();
}

int32_t ReaderImpl::Init(const std::string& topic, uint32_t capcaticy) {
    _topic = topic;
    _queue = Topology::GetInstance().CreateRecvQueue(topic, capcaticy);
    // ADF_INTERNAL_LOG_DEBUG << "Create reader of topic " << topic << ", capacity " << capcaticy;

    return 0;
}

int32_t ReaderImpl::Init(const std::string& topic, Reader::Callback cb, uint32_t capcaticy) {
    _topic = topic;
    _callback = cb;
    _queue = Topology::GetInstance().CreateRecvQueue(topic, capcaticy);
    _recv_thread = std::make_shared<std::thread>([this](){
        while (!_need_stop) {
            auto ptr = GetLatestOneBlocking(10000, true);
            if (ptr) {
                if (_callback) {
                    _callback(ptr);
                }
            }
        }
    });
    // ADF_INTERNAL_LOG_DEBUG << "Create reader with callback of topic " << topic << ", capacity " << capcaticy;
    
    return 0;
}

void ReaderImpl::Pause() {
    _queue->EnableWrite(false);
}

void ReaderImpl::Resume() {
    _queue->EnableWrite(true);
}

BaseDataTypePtr ReaderImpl::GetLatestOneBlocking(const uint32_t timeout_ms, bool erase) {
    auto ptr = _queue->GetLatestOneBlocking(erase, timeout_ms);
    if (ptr) {
        return *ptr;
    }

    return nullptr;
}

BaseDataTypePtr ReaderImpl::GetLatestOne(bool erase) {
    auto ptr = _queue->GetLatestOne(erase);
    if (ptr) {
        return *ptr;
    }

    return nullptr;
}

std::vector<BaseDataTypePtr> ReaderImpl::GetLatestNdata(const size_t n, bool erase) {
    std::vector<BaseDataTypePtr> vec;
    std::vector<std::shared_ptr<BaseDataTypePtr>> ret = _queue->GetLatestNdata(n, erase);

    for (auto& ptr : ret) {
        vec.emplace_back(*ptr);
    }

    return vec;
}

}
}
}