#pragma once

#include "cm/include/proxy.h"
#include "idl/generated/cm_protobuf.h"
#include "idl/generated/cm_protobufPubSubTypes.h"

namespace hozon {
namespace netaos {
namespace cm {

template<typename MessageT>
class ProtoCMReader {
public:
    ProtoCMReader() : 
            _proxy(std::make_shared<CmProtoBufPubSubType>()) {

    }

    ~ProtoCMReader() {

    }

    using CallbackFunc = std::function<void(std::shared_ptr<MessageT>)>;
    int32_t Init(const uint32_t domain, const std::string& topic, CallbackFunc cb) {
        _cb = cb;
        int32_t ret = _proxy.Init(domain, topic);
        if (ret < 0) {
            return ret;
        }

        _proxy.Listen(std::bind(&ProtoCMReader<MessageT>::ProxyListenCallback, this));

        return 0;
    }

    void Deinit() {
        _proxy.Deinit();
    }

private:
    void ProxyListenCallback(void) {
        std::shared_ptr<CmProtoBuf> cm_pb(new CmProtoBuf);
        _proxy.Take(cm_pb);

        std::shared_ptr<MessageT> msg(new MessageT);
        msg->ParseFromArray(cm_pb->str().data(), cm_pb->str().size());

        _cb(msg);
    }

    Proxy _proxy;
    CallbackFunc _cb;
};

}
}
}