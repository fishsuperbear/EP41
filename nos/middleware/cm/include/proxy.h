#pragma once

#include <fastdds/dds/topic/TopicDataType.hpp>

namespace hozon {
namespace netaos {
namespace cm {

class ProxyImpl;

class Proxy {
   public:
    Proxy(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> topic_type);
    Proxy(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> topic_type, uint8_t qos_mode);
    ~Proxy();
    int32_t Init(const uint32_t domain, const std::string& topic);
    void Deinit();

    int32_t Take(std::shared_ptr<void> data, uint64_t timeout_us);
    int32_t Take(std::shared_ptr<void> data);

    template <class T>
    int32_t Take(const std::function<void(const T&)>& do_take);

    using DataAvailableCallback = std::function<void(void)>;
    void Listen(DataAvailableCallback callback);

    bool IsMatched();

   private:
    std::shared_ptr<ProxyImpl> _pimpl;
};

}  // namespace cm
}  // namespace netaos
}  // namespace hozon
