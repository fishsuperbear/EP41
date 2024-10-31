#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace hozon {
namespace netaos {
namespace adf {

using ITCDataType = std::shared_ptr<void>;

class ITCWriterImpl;
class ITCReaderImpl;

class ITCWriter {
   public:
    ITCWriter();
    ~ITCWriter();

    int32_t Init(const std::string& topic_name);
    void Deinit();

    void Write(ITCDataType data);

   private:
    std::unique_ptr<ITCWriterImpl> _pimpl;
};

class ITCReader {
   public:
    ITCReader();
    ~ITCReader();

    // Init by polling mode, the oldest data will be overwritten when capacity is exceeded
    int32_t Init(const std::string& topic_name, uint32_t capacity = 5);

    // Init by callback mode, never call any "Take" in callback
    using CallbackFunc = std::function<void(ITCDataType)>;
    int32_t Init(const std::string& topic_name, CallbackFunc callback, uint32_t capacity = 5);

    // Stop to receive
    void Deinit();

    // Take latest one without waiting
    ITCDataType Take();

    // Take latest one with timeout
    ITCDataType Take(const uint32_t timeout_ms);

   private:
    std::unique_ptr<ITCReaderImpl> _pimpl;
};

}  // namespace adf
}  // namespace netaos
}  // namespace hozon