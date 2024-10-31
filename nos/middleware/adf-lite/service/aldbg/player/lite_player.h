

#ifndef LITE_PLAYER_H_
#define LITE_PLAYER_H_

#include "middleware/adf-lite/include/cm_reader.h"
#include "adf-lite/include/adf_lite_internal_logger.h"
#include "adf-lite/include/topology.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

using namespace eprosima::fastdds::dds;
using namespace eprosima::fastrtps;
class LitePlayer {
   public:
    LitePlayer():_status(false) {};
    virtual ~LitePlayer() {    
        End();
    };
    void Start(const std::vector<RoutingAttr>& attr);
    void End();
private:
    void ReceiveTopic(const std::string& topic, std::shared_ptr<CmProtoBuf> data);
    void SubscribeLiteTopic(const std::string topic);
    bool SubscribeLiteTopics(const std::vector<RoutingAttr>& _routing_attrs);
    void DesubscribeLiteTopics();
    BaseDataTypePtr CreateBaseDataFromProto(std::shared_ptr<google::protobuf::Message> msg);
    std::string _process_name;
    std::map<std::string, std::shared_ptr<hozon::netaos::adf_lite::CMReader>> _readers;
    bool _status;
};

}  // namespace adf_lite
}  // namespace netaos
}  // namespace hozon

#endif /* LITE_PLAYER_H_ */