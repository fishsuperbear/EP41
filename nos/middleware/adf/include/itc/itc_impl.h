#pragma once

#include "adf/include/itc/itc.h"
#include "adf/include/itc/itc_cv_queue.h"

#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

namespace hozon {
namespace netaos {
namespace adf {

class ITCTopic;

using ITCQueueType = hozon::netaos::adf::CVSizeLimitQueue<ITCDataType>;

class ITCWriterImpl {
   public:
    int32_t Init(const std::string& topic_name);
    void Deinit();

    void Write(ITCDataType data);

   private:
    ITCTopic* _itc_topic;
};

class ITCReaderImpl {
   public:
    int32_t Init(const std::string& topic_name, uint32_t capacity = 5);
    int32_t Init(const std::string& topic_name, ITCReader::CallbackFunc callback, uint32_t capacity = 5);

    // Stop to receive
    void Deinit();

    // Take without waiting
    ITCDataType Take();

    // Take with timeout
    ITCDataType Take(const uint32_t timeout_ms);

   private:
    void PollRoutine();

    ITCTopic* _itc_topic;
    std::shared_ptr<ITCQueueType> _queue;
    ITCReader::CallbackFunc _cb = nullptr;
    std::shared_ptr<std::thread> _poll_thread = nullptr;
    bool _need_stop = false;
};

class ITCTopic {
   public:
    void AddReaderQueue(ITCQueueType* reader_queue) {
        _mtx.lock();
        _reader_queues.emplace_back(reader_queue);
        _mtx.unlock();
    }

    void DeleteReaderQueue(ITCQueueType* reader_queue) {
        _mtx.lock();
        for (auto it = _reader_queues.begin(); it != _reader_queues.end(); ++it) {
            if (*it == reader_queue) {
                _reader_queues.erase(it);
                _mtx.unlock();
                return;
            }
        }
        _mtx.unlock();
    }

    void WriteToAll(ITCDataType data) {
        _mtx.lock();
        for (auto& reader_queue : _reader_queues) {
            reader_queue->PushOneAndNotify(data);
        }
        _mtx.unlock();
    }

   private:
    std::mutex _mtx;
    std::vector<ITCQueueType*> _reader_queues;
};

class ITC {
   public:
    static ITC& GetInstance() {
        static ITC instance;

        return instance;
    }

    ITCTopic* GetOrCreateTopic(const std::string& topic_name) {
        ITCTopic* itc_topic = nullptr;

        _mtx.lock();
        itc_topic = &_topic_map[topic_name];
        _mtx.unlock();

        return itc_topic;
    }

   private:
    ITC() {}

    std::mutex _mtx;
    std::unordered_map<std::string, ITCTopic> _topic_map;
};

}  // namespace adf
}  // namespace netaos
}  // namespace hozon