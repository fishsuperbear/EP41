#include "cm/include/proxy.h"
#include "idl/generated/cm_protobuf.h"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "proto/dead_reckoning/dr.pb.h"
#include "proto/localization/localization.pb.h"


namespace hozon {
namespace ethstack {
namespace lidar {

class ProtoDRReader{
public:
    ProtoDRReader():_proxy(std::make_shared<CmProtoBufPubSubType>()){
    }

    ~ProtoDRReader(){
    }

    int32_t Init(const uint32_t domain,const std::string& topic){
        int32_t ret = _proxy.Init(domain,topic);
        if (ret < 0){
            return ret;
        }
        return 0;  
    }

    void Deinit(){
        _proxy.Deinit();
    }

    // std::shared_ptr<hozon::dead_reckoning::DeadReckoning> GetDeadReckoning(){
    //     std::shared_ptr<CmProtoBuf> cm_pb(new CmProtoBuf);
    //     isGetFlag = _proxy.Take(cm_pb);

    //     std::shared_ptr<hozon::dead_reckoning::DeadReckoning> msg(new hozon::dead_reckoning::DeadReckoning);
    //     msg->ParseFromArray(cm_pb->str().data(), cm_pb->str().size());
    //     return msg;
    // }
    
    std::shared_ptr<hozon::localization::Localization> GetDeadReckoning(){
        std::shared_ptr<CmProtoBuf> cm_pb(new CmProtoBuf);
        isGetFlag = _proxy.Take(cm_pb);

        std::shared_ptr<hozon::localization::Localization> msg(new hozon::localization::Localization);
        msg->ParseFromArray(cm_pb->str().data(), cm_pb->str().size());
        return msg;
    }

    int32_t isGetData(){
        return isGetFlag;
    }


    
private:
    hozon::netaos::cm::Proxy _proxy;
    int32_t isGetFlag;

};

}
}
}