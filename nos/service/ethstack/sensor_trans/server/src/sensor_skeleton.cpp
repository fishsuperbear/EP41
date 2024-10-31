#include <memory>
#include "sensor_skeleton.h"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "logger.h"

namespace hozon {
namespace netaos {
namespace sensor {

Skeleton::Skeleton(uint32_t domainID, std::string topic) {
    skeleton_ = std::make_shared<cm::Skeleton>(std::make_shared<CmProtoBufPubSubType>());
    skeleton_->Init(domainID, topic);
    SENSOR_LOG_INFO << "dominaID " << domainID << ", topic " << topic << " skeleton."; 
}
Skeleton::~Skeleton() { }
int Skeleton::Write(std::shared_ptr<void> data) {
    if(0 == skeleton_->Write(data)) { 
        return 0;
    }
    return -1;
}

int Skeleton::Deinit() {
    skeleton_->Deinit();
    SENSOR_LOG_INFO << "sensor skeleton Deinit sussessful."; 
    return 0;
}

}   // namespace sensor
}   // namespace netaos
}   // namespace hozon