#include "dds_player_publisher.h"
#include <unordered_map>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <google/protobuf/message.h>
#include "bag_data_pubsubtype.h"
#include "cm/include/cm_config.h"
#include "cost_calc.h"
#include "data_tools_logger.hpp"
#include "fastdds/dds/publisher/Publisher.hpp"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "idl/generated/cm_protobufTypeObject.h"
#include "idl/generated/cm_someipbufTypeObject.h"
#include "idl/generated/proto_methodTypeObject.h"
#include "proto_factory.h"
#include "proto_utility.h"
#include "topic_manager.hpp"
#include "proto/soc/point_cloud.pb.h"
#include "proto/soc/sensor_image.pb.h"

#define UNUSED_VAR(var) (void)(var)

namespace hozon {

namespace netaos {
namespace bag {

using namespace eprosima::fastrtps::rtps;
using namespace eprosima::fastrtps;

DDSPlayerPublisher::~DDSPlayerPublisher(){

};

bool DDSPlayerPublisher::InitWriters(std::map<std::string, std::string> topicTypeMap, const bool play_protomethod, bool with_original_h265) {

    stopped_ = false;
    with_original_h265_ = with_original_h265;
    // Create different participants and publishers by topics.
    // Participants qos differs in lidar, yuv, h265, normal, sd.
    bool has_lidar = false;
    bool has_camera_h265 = false;
    bool has_camera_yuv = false;

    for (auto& topic : topicTypeMap) {
        int32_t data_type = topic_manager_->GetTopicDataType(topic.first);
        switch (data_type) {
            case hozon::netaos::data_tool_common::kDdsDataType_SD: {
                // never in this case.
            } break;
            case hozon::netaos::data_tool_common::kDdsDataType_Normal: {

            } break;
            case hozon::netaos::data_tool_common::kDdsDataType_Lidar: {
                has_lidar = true;
            } break;
            case hozon::netaos::data_tool_common::kDdsDataType_LidarRaw: {
                // raw point cloud is always transferred to point cloud.
                has_lidar = true;
            } break;
            case hozon::netaos::data_tool_common::kDdsDataType_CameraYuv: {
                // This case only in H265PlayHandler
            } break;
            case hozon::netaos::data_tool_common::kDdsDataType_CameraH265:
                if (with_original_h265) {
                    // Only for play camera in h265 data.
                    has_camera_h265 = true;
                } else {
                }
                break;
        }
    }

    // Craete participant for subscribing normal topics.
    auto default_pub_participant = topic_manager_->GetParticipant(hozon::netaos::data_tool_common::kDdsDataType_Normal, true);
    auto publisher = topic_manager_->GetPublisher(hozon::netaos::data_tool_common::kDdsDataType_Normal);

    std::shared_ptr<eprosima::fastdds::dds::DomainParticipant> lidar_pub_participant;
    if (has_lidar) {
        BAG_LOG_DEBUG << "create particpant for lidar";
        lidar_pub_participant = topic_manager_->GetParticipant(hozon::netaos::data_tool_common::kDdsDataType_Lidar, true);
        auto publisher = topic_manager_->GetPublisher(hozon::netaos::data_tool_common::kDdsDataType_Lidar);
        UNUSED_VAR(publisher);
    }

    std::shared_ptr<eprosima::fastdds::dds::DomainParticipant> camera_h265_pub_participant;
    if (has_camera_h265) {
        BAG_LOG_DEBUG << "create particpant for lidar h265";
        camera_h265_pub_participant = topic_manager_->GetParticipant(hozon::netaos::data_tool_common::kDdsDataType_CameraH265, true);
        auto publisher = topic_manager_->GetPublisher(hozon::netaos::data_tool_common::kDdsDataType_CameraH265);
        UNUSED_VAR(publisher);
    }

    //regist  CmProtoBuf
    registercm_protobufTypes();
    std::shared_ptr<HelloWorldPubSubType> cmProtoBuf_ptr = std::make_shared<HelloWorldPubSubType>();
    cmProtoBuf_ptr->setName("CmProtoBuf");
    eprosima::fastdds::dds::TypeSupport cmProtoBuf_type(cmProtoBuf_ptr);
    cmProtoBuf_type.get()->auto_fill_type_information(true);
    cmProtoBuf_type.get()->auto_fill_type_object(false);
    if (default_pub_participant) {
        cmProtoBuf_type.register_type(default_pub_participant.get());
    }
    if (lidar_pub_participant) {
        cmProtoBuf_type.register_type(lidar_pub_participant.get());
    }
    if (camera_h265_pub_participant) {
        cmProtoBuf_type.register_type(camera_h265_pub_participant.get());
    }

    //regist  CmSomeipBuf
    registercm_someipbufTypes();
    std::shared_ptr<HelloWorldPubSubType> cmSomeipBuf_ptr = std::make_shared<HelloWorldPubSubType>();
    cmSomeipBuf_ptr->setName("CmSomeipBuf");
    eprosima::fastdds::dds::TypeSupport cmSomeipBuf_type(cmSomeipBuf_ptr);
    cmSomeipBuf_type.get()->auto_fill_type_information(true);
    cmSomeipBuf_type.get()->auto_fill_type_object(false);
    if (default_pub_participant) {
        cmSomeipBuf_type.register_type(default_pub_participant.get());
    }

    if (play_protomethod) {
        //regist  protomethod
        registerproto_methodTypes();
        std::shared_ptr<HelloWorldPubSubType> protomethod_ptr = std::make_shared<HelloWorldPubSubType>();
        protomethod_ptr->setName("ProtoMethodBase");
        eprosima::fastdds::dds::TypeSupport protomethod_type(protomethod_ptr);
        protomethod_type.get()->auto_fill_type_information(true);
        protomethod_type.get()->auto_fill_type_object(false);
        protomethod_type.register_type(default_pub_participant.get());
    }

    bool h265_to_yuv_play = false;
    for (auto item : topicTypeMap) {
        //REGISTER THE TYPE
        if (!("CmProtoBuf" == item.second || "CmSomeipBuf" == item.second || (play_protomethod && "ProtoMethodBase" == item.second))) {
            BAG_LOG_DEBUG << "type:" << item.second << " can't be played! not supported now!";
            continue;
        }

        int32_t data_type = topic_manager_->GetTopicDataType(item.first);
        if (data_type == hozon::netaos::data_tool_common::kDdsDataType_LidarRaw) {
            // raw point cloud is transferred to point cloud. modify the data type to use the rigth participant and publisher.
            data_type = hozon::netaos::data_tool_common::kDdsDataType_Lidar;
        } else if ((data_type == hozon::netaos::data_tool_common::kDdsDataType_CameraH265) && !with_original_h265_) {
            h265_to_yuv_play = true;
            continue;
        } else {
        }

        auto participant = topic_manager_->GetParticipant(data_type, true);
        auto publisher = topic_manager_->GetPublisher(data_type);
        Topic* topic = participant->create_topic(item.first, item.second, TOPIC_QOS_DEFAULT);

        if (topic == nullptr) {
            return false;
        }
        topiclist_.push_back(topic);

        // CREATE THE WRITER
        DataWriterQos wqos = topic_manager_->GetWriterQos(data_type, item.first);

        DataWriter* writer = publisher->create_datawriter(topic, wqos);
        if (nullptr == writer) {
            BAG_LOG_ERROR << "create DataWriter failed! topic:" << item.first << " type:" << item.second;
        }

        _topicWriterMap[item.first] = writer;
        BAG_LOG_INFO << "ready to play topic:" << item.first << " type:" << item.second;
    }

    // h265_to_yuv is processed in h265_player_publisher
    if ((0 == _topicWriterMap.size()) && !h265_to_yuv_play) {
        BAG_LOG_WARN << "no valid message to play";
        return false;
    }

    return true;
};

bool DDSPlayerPublisher::DeinitWriters() {
    stopped_ = true;

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    for (auto item : _topicWriterMap) {
        int32_t data_type = topic_manager_->GetTopicDataType(item.first);
        if (data_type == hozon::netaos::data_tool_common::kDdsDataType_LidarRaw) {
            // raw point cloud is transferred to point cloud. modify the data type to use the rigth participant and publisher.
            data_type = hozon::netaos::data_tool_common::kDdsDataType_Lidar;
        }

        if ((data_type == hozon::netaos::data_tool_common::kDdsDataType_CameraH265) && !with_original_h265_) {
            continue;
        }

        auto publisher = topic_manager_->GetPublisher(data_type);

        // const auto* publisher = item.second->get_publisher();

        if (publisher && (nullptr != item.second)) {
            publisher->delete_datawriter(item.second);
        }
    }

    for (auto topic : topiclist_) {
        std::string temp = topic->get_name();
        int32_t data_type = topic_manager_->GetTopicDataType(topic->get_name());
        if (data_type == hozon::netaos::data_tool_common::kDdsDataType_LidarRaw) {
            // raw point cloud is transferred to point cloud. modify the data type to use the rigth participant and publisher.
            data_type = hozon::netaos::data_tool_common::kDdsDataType_Lidar;
        }
        if ((data_type == hozon::netaos::data_tool_common::kDdsDataType_CameraH265) && !with_original_h265_) {
            continue;
        }
        auto participant = topic_manager_->GetParticipant(data_type, true);
        // const auto* participant = topic->get_participant();
        if (topic != nullptr) {
            participant->delete_topic(topic);
        }
    }

    return true;
}

void DDSPlayerPublisher::Publish(BagMessage* bagMessage) {
    if (stopped_) {
        return;
    }
    if (_topicWriterMap.find(bagMessage->topic) == _topicWriterMap.end()) {
        // BAG_LOG_DEBUG << "publish message failed! no writer for topic:" << bagMessage->topic;
        return;
    }
    DataWriter* mp_writer = _topicWriterMap[bagMessage->topic];

    if (update_pubtime_ && bagMessage->type == "CmProtoBuf") {
        BagDataType data;
        {
            // COST_CALC_OP("update pubtime");
            // Deserialize to idl type.
            CmProtoBuf idl_message;
            CmProtoBufPubSubType sub_type;
            if (sub_type.deserialize(bagMessage->data.m_payload.get(), &idl_message)) {

                // Deserialize to proto type.
                std::string proto_name = idl_message.name();
                std::shared_ptr<google::protobuf::Message> proto_message;
                if (proto_name.find("PointCloud") != std::string::npos) {
                    proto_message = std::shared_ptr<google::protobuf::Message>(new hozon::soc::PointCloud());
                } else if (proto_name.find("CompressedImage") != std::string::npos) {
                    proto_message = std::shared_ptr<google::protobuf::Message>(new hozon::soc::CompressedImage());
                } else if (proto_name.find("Image") != std::string::npos) {
                    proto_message = std::shared_ptr<google::protobuf::Message>(new hozon::soc::Image());
                } else {
                    proto_message = std::shared_ptr<google::protobuf::Message>(hozon::netaos::data_tool_common::ProtoFactory::getInstance()->GenerateMessageByType(proto_name));
                }

                if (proto_message == NULL) {
                    std::cerr << "Cannot create prototype message from message descriptor";
                    return;
                }
                if (proto_message->ParseFromArray(idl_message.str().data(), idl_message.str().size())) {

                    // Find header / publish_stamp
                    if (hozon::netaos::data_tool_common::ProtoUtility::HasProtoField(*proto_message, "header")) {

                        google::protobuf::Message& header_msg = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionMutableMessage(*proto_message, "header");
                        double pub_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
                        if (hozon::netaos::data_tool_common::ProtoUtility::SetProtoReflectionDouble(header_msg, "publish_stamp", pub_time)) {}

                        std::string buf;
                        if (proto_message->SerializeToString(&buf)) {
                            idl_message.str().resize(buf.size());
                            memcpy(idl_message.str().data(), buf.data(), buf.size());
                            if (sub_type.serialize(&idl_message, bagMessage->data.m_payload.get())) {}
                        }
                    }
                }
            }
        }

        // It is not a regulare proto message, just publish as it.
        data.m_payload->copy(bagMessage->data.m_payload.get(), false);
        mp_writer->write(&data);
    } else {
        mp_writer->write(&bagMessage->data);
    }
};

void DDSPlayerPublisher::SetUpdatePubTimeFlag(bool flag) {
    update_pubtime_ = flag;
}

}  // namespace bag
}  //namespace netaos
}  //namespace hozon