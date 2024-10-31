#include "adf/include/itc/itc_impl.h"

namespace hozon {
namespace netaos {
namespace adf {

int32_t ITCWriterImpl::Init(const std::string& topic_name) {
    _itc_topic = ITC::GetInstance().GetOrCreateTopic(topic_name);

    return 0;
}

void ITCWriterImpl::Deinit() {
    return;
}

void ITCWriterImpl::Write(ITCDataType data) {
    _itc_topic->WriteToAll(data);
}

int32_t ITCReaderImpl::Init(const std::string& topic_name, uint32_t capacity) {
    _queue = std::make_shared<ITCQueueType>(capacity);

    _itc_topic = ITC::GetInstance().GetOrCreateTopic(topic_name);
    _itc_topic->AddReaderQueue(_queue.get());

    return 0;
}

int32_t ITCReaderImpl::Init(const std::string& topic_name, ITCReader::CallbackFunc callback, uint32_t capacity) {
    _cb = callback;
    int32_t ret = Init(topic_name, capacity);
    if (ret < 0) {
        return -1;
    }

    _poll_thread = std::make_shared<std::thread>(&ITCReaderImpl::PollRoutine, this);

    return 0;
}

void ITCReaderImpl::Deinit() {
    _itc_topic->DeleteReaderQueue(_queue.get());
    _need_stop = true;
    _queue->Exit();
    if (_poll_thread) {
        _poll_thread->join();
    }
}

ITCDataType ITCReaderImpl::Take() {
    auto ret = _queue->GetLatestOne(true);
    return ret ? *ret : nullptr;
}

ITCDataType ITCReaderImpl::Take(const uint32_t timeout_ms) {
    auto ret = _queue->GetLatestOneBlocking(true, timeout_ms);
    return ret ? *ret : nullptr;
}

void ITCReaderImpl::PollRoutine() {
    while (!_need_stop) {
        ITCDataType ret = Take(UINT32_MAX);
        if (ret) {
            if (_cb) {
                _cb(ret);
            }
        }
    }
}

}  // namespace adf
}  // namespace netaos
}  // namespace hozon