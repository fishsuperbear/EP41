#include "cm/include/method.h"
#include "idl/generated/proto_method.h"
#include "idl/generated/proto_methodPubSubTypes.h"

namespace hozon {
namespace netaos {
namespace cm {

template<typename ReqProto, typename RespProto>
class ProtoMethodClient {
public:
    ProtoMethodClient() :
            _cm_client(std::make_shared<ProtoMethodBasePubSubType>(), std::make_shared<ProtoMethodBasePubSubType>()) {
        
    }

    int32_t Init(const uint32_t domain, const std::string& service_name) {
        return _cm_client.Init(domain, service_name);
    }

    int32_t DeInit() {
        return _cm_client.Deinit();
    }

    int32_t WaitServiceOnline(int64_t timeout_ms) {
        return _cm_client.WaitServiceOnline(timeout_ms);
    }

    int32_t Request(std::shared_ptr<ReqProto> req, std::shared_ptr<RespProto> resp, int64_t timeout_ms) {
        std::shared_ptr<ProtoMethodBase> idl_req(new ProtoMethodBase);
        std::shared_ptr<ProtoMethodBase> idl_resp(new ProtoMethodBase);

        idl_req->name(resp->GetTypeName());
        std::string serialized_string;
        req->SerializeToString(&serialized_string);
        idl_req->str().assign(serialized_string.begin(), serialized_string.end());
        
        int32_t ret = _cm_client.Request(idl_req, idl_resp, timeout_ms);
        if (ret < 0) {
            return ret;
        }

        bool bret = resp->ParseFromArray(idl_resp->str().data(), idl_resp->str().size());
        if (!bret) {
            return -9;
        }

        return 0;
    }

    std::future<std::pair<int32_t, std::shared_ptr<RespProto>>> AsyncRequest(std::shared_ptr<ReqProto> req, int64_t timeout_ms) {
        return std::async(std::launch::async, [this, req, timeout_ms] {
            std::shared_ptr<RespProto> resp = std::make_shared<RespProto>();
            int32_t ret = Request(req, resp, timeout_ms);
            std::pair<int32_t, std::shared_ptr<RespProto>> pack(ret, resp);
            return pack;
        });
    }

    int32_t RequestAndForget(std::shared_ptr<ReqProto> req) {
        return _cm_client.RequestAndForget(req);
    }

private:
    Client<ProtoMethodBase, ProtoMethodBase> _cm_client;
};

template <typename ReqProto, typename RespProto>
class ProtoMethodServer : public Server<ProtoMethodBase, ProtoMethodBase> {
public:
    using Callback = std::function<int32_t(const std::shared_ptr<ReqProto>&, std::shared_ptr<RespProto>&)>;
    ProtoMethodServer(Callback cb) :
            Server<ProtoMethodBase, ProtoMethodBase>(std::make_shared<ProtoMethodBasePubSubType>(), std::make_shared<ProtoMethodBasePubSubType>()) {
        _cb = cb;
    }

    virtual int32_t Process(const std::shared_ptr<ProtoMethodBase> idl_req, std::shared_ptr<ProtoMethodBase> idl_resp) override {
        std::shared_ptr<ReqProto> proto_req(new ReqProto);

        bool ret = proto_req->ParseFromArray(idl_req->str().data(), idl_req->str().size());
        if (!ret) {
            return -1;
        }

        std::shared_ptr<RespProto> proto_resp(new RespProto);
        int32_t iret = _cb(proto_req, proto_resp);
        if (iret < 0) {
            return iret;
        }

        idl_resp->name(proto_resp->GetTypeName());
        std::string serialized_string;
        proto_resp->SerializeToString(&serialized_string);
        idl_resp->str().assign(serialized_string.begin(), serialized_string.end());

        return 0;
    }

private:
    Callback _cb;
};

}
}    
}