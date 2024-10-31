#pragma once
#include "cm/include/proxy.h"
#include "idl/generated/cm_protobuf.h"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "adf-lite/include/adf_lite_internal_logger.h"

using namespace hozon::netaos::cm;
namespace hozon {
namespace netaos {
namespace adf_lite {

class CMReader {
public:
    CMReader() : 
        _proxy(std::make_shared<CmProtoBufPubSubType>()) {
    }
    ~CMReader() {}

    using CallbackFunc = std::function<void(const std::string&, std::shared_ptr<CmProtoBuf>)>;
    int32_t Init(const uint32_t domain, const std::string& topic, CallbackFunc cb) {
        _cb = cb;
        int32_t ret = _proxy.Init(domain, topic);
        if (ret < 0) {
            ADF_INTERNAL_LOG_ERROR << "proxy Init failed";
            return ret;
        }
        _topic = topic;
        _proxy.Listen(std::bind(&CMReader::ProxyListenCallback, this));
        return 0;
    }

    void Deinit() {
        _proxy.Deinit();
    }

private:

    void ProxyListenCallback(void) {
        std::shared_ptr<CmProtoBuf> cm_pb(new CmProtoBuf);
        _proxy.Take(cm_pb);
        _cb(_topic, cm_pb);
    }

    Proxy _proxy;
    CallbackFunc _cb;
    std::string _topic;
};

}
}
}