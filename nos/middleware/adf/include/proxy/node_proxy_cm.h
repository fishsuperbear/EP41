#pragma once
#include "adf/include/node_proxy.h"
#include "cm/include/proxy.h"

namespace hozon {
namespace netaos {
namespace adf {
#define PROXY_INIT(proxy_ptr, pub_sub_type, domain, topic, receive)                                \
    {                                                                                              \
        proxy_ptr = std::make_shared<hozon::netaos::cm::Proxy>(pub_sub_type);                      \
        if (0 == proxy_ptr->Init(domain, topic)) {                                                 \
            proxy_ptr->Listen(std::bind(&receive, this));                                          \
        } else {                                                                                   \
            ADF_LOG_ERROR << "Init damain ( " << domain << " ), topic ( " << topic << " ) fail !"; \
        }                                                                                          \
    }

#define PROXY_DEINIT(proxy_ptr) \
    { proxy_ptr->Deinit(); }

class NodeProxyCM : public NodeProxyBase {
   public:
    explicit NodeProxyCM(const NodeConfig::CommInstanceConfig& config,
                         std::shared_ptr<eprosima::fastdds::dds::TopicDataType> pub_sub_type);
    ~NodeProxyCM();
    void PauseReceive() override;
    void ResumeReceive() override;
    void Deinit() override;
    virtual void OnDataReceive(void) override;

   protected:
    BaseDataTypePtr CreateBaseDataFromIDL(std::shared_ptr<IDLBaseType> idl_msg);

    /* data */
    std::shared_ptr<eprosima::fastdds::dds::TopicDataType> _pub_sub_type;
    std::shared_ptr<hozon::netaos::cm::Proxy> _proxy;
    uint32_t _domain;
    std::string _topic;
};
}  // namespace adf
}  // namespace netaos
}  // namespace hozon
