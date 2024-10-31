#include "adf-lite/service/aldbg/utility/utility.h"
#include "adf-lite/service/aldbg/player/lite_player.h"
#include "adf-lite/include/cm_reader.h"
#include "adf-lite/include/topology.h"
#include "adf/include/node_proto_register.h"
#include "adf-lite/service/rpc/lite_rpc.h"
#include "adf-lite/include/struct_register.h"
#include "adf-lite/include/writer.h"

using namespace hozon::netaos::adf;
namespace hozon {
namespace netaos {
namespace adf_lite {
void LitePlayer::Start(const std::vector<RoutingAttr>& attr) {
    if (!_status) {
        _status = true;
        auto func = LiteRpc::GetInstance().GetStringServiceFunc("GetProcessName");
        if (func == nullptr) {
            ADF_INTERNAL_LOG_WARN << "GetProcessName Func maybe not beed registed";
        } else {
            int32_t res = func(_process_name);
            if (res < 0) {
                ADF_INTERNAL_LOG_WARN << "GetProcessName is Has Error";
            }
        }
        SubscribeLiteTopics(attr);
    } else {
    }
}

void LitePlayer::End() {
    if (_status) {
        DesubscribeLiteTopics();
        _status = false;
    } else {
    }
}

BaseDataTypePtr LitePlayer::CreateBaseDataFromProto(std::shared_ptr<google::protobuf::Message> msg) {
    BaseDataTypePtr base_ptr(new BaseData);
    base_ptr->proto_msg = msg;

    return base_ptr;
}

void LitePlayer::ReceiveTopic(const std::string &topic, std::shared_ptr<CmProtoBuf> data)
{
    std::string inner_topic;
    int32_t ret = GetInnerTopicFromCmTopic(topic, inner_topic);
    if (ret != 0) {
        ADF_INTERNAL_LOG_DEBUG << "illegal lite topic :[" << topic << "]";
        return;
    }

    if (BaseDataTypeMgr::GetInstance().GetSize(inner_topic) != 0) {
        ADF_INTERNAL_LOG_VERBOSE << "Parse as Registered struct!";
        std::string serialize_data;
        serialize_data.resize(data->str().size());
        copy(data->str().begin(), data->str().end(), serialize_data.begin());

        BaseDataTypePtr pub_data = BaseDataTypeMgr::GetInstance().Create(inner_topic);
        if (pub_data == nullptr) {
            ADF_INTERNAL_LOG_ERROR << "Create struct type [" << inner_topic << "] failed, class type is [" << data->name() << "]";
            return;
        }

        pub_data->ParseFromString(serialize_data, BaseDataTypeMgr::GetInstance().GetSize(inner_topic));
        ADF_INTERNAL_LOG_DEBUG << "Send topic:[" << inner_topic << "] to Topology";
        Writer _writer;
        _writer.Init(inner_topic);
        _writer.Write(pub_data);
        return;
    }

    ADF_INTERNAL_LOG_VERBOSE << "Parse as proto message!";
    std::shared_ptr<google::protobuf::Message> proto_msg = ProtoMessageTypeMgr::GetInstance().Create(inner_topic);
    if (proto_msg == nullptr) {
        ADF_INTERNAL_LOG_ERROR << "Create proto type [" << inner_topic << "] failed, class type is [" << data->name() << "]";
        return;
    }

    ret = proto_msg->ParseFromArray(data->str().data(), data->str().size());
    if (!ret) {
        ADF_INTERNAL_LOG_ERROR << "ParseFromArray has error";
        return;
    }
    BaseDataTypePtr pub_data = CreateBaseDataFromProto(proto_msg);

    Writer _writer;
    _writer.Init(inner_topic);
    _writer.Write(pub_data);
}

void LitePlayer::SubscribeLiteTopic(const std::string topic) {
    std::string cm_topic = "/lite/" + _process_name + "/event/" + topic;

    if (_readers.find(topic) == _readers.end()) {
        ADF_INTERNAL_LOG_INFO << "Subscribe topic:" << topic;
        std::shared_ptr<hozon::netaos::adf_lite::CMReader> reader = std::make_shared<hozon::netaos::adf_lite::CMReader>();
        reader->Init(0, cm_topic, std::bind(&LitePlayer::ReceiveTopic, this, std::placeholders::_1, std::placeholders::_2));
        _readers[topic] = reader;
    } else {
        ADF_INTERNAL_LOG_INFO << "topic:" << topic << "has been in _readers";
    }
}

bool LitePlayer::SubscribeLiteTopics(const std::vector<RoutingAttr>& _routing_attrs) {
    for (auto& attr : _routing_attrs) {
        SubscribeLiteTopic(attr.topic);
    }

    return true;
}

void LitePlayer::DesubscribeLiteTopics() {
    for (auto& reader : _readers) {
        if (reader.second != nullptr) {
            reader.second->Deinit();
        }
    }
}
}  // namespace adf_lite
}  // namespace netaos
}  // namespace hozon
