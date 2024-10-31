#include "cm/include/proxy.h"
#include "cm/include/proxy_impl.h"

#include "idl/generated/zerocopy_imagePubSubTypes.h"

namespace hozon {
namespace netaos {
namespace cm {

Proxy::Proxy(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> topic_type) : _pimpl(std::make_shared<ProxyImpl>(topic_type)) {}

Proxy::Proxy(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> topic_type, uint8_t qos_mode) : _pimpl(std::make_shared<ProxyImpl>(topic_type, qos_mode)) {}

Proxy::~Proxy() {}

int32_t Proxy::Init(const uint32_t domain, const std::string& topic) {
    return _pimpl->Init(domain, topic);
}

void Proxy::Deinit() {
    return _pimpl->Deinit();
}

int32_t Proxy::Take(std::shared_ptr<void> data) {
    return _pimpl->Take(data);
}

int32_t Proxy::Take(std::shared_ptr<void> data, uint64_t timeout_us) {
    return _pimpl->Take(data, timeout_us);
}

template <class T>
int32_t Proxy::Take(const std::function<void(const T&)>& do_take) {
    return _pimpl->Take<T>(do_take);
}

void Proxy::Listen(DataAvailableCallback callback) {
    _pimpl->Listen(callback);
}

bool Proxy::IsMatched() {
    return _pimpl->IsMatched();
}

template int32_t Proxy::Take<ZeroCopyImg8M420>(const std::function<void(const ZeroCopyImg8M420&)>&);
template int32_t Proxy::Take<ZeroCopyImg2M422>(const std::function<void(const ZeroCopyImg2M422&)>&);

}  // namespace cm
}  // namespace netaos
}  // namespace hozon
