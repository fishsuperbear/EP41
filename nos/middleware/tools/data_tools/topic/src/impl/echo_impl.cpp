
#include "impl/echo_impl.h"
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <unordered_map>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/DomainParticipantListener.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include <fastrtps/attributes/SubscriberAttributes.h>
#include <fastrtps/rtps/common/Types.h>
#include <fastrtps/subscriber/SampleInfo.h>
#include <fastrtps/types/DynamicDataFactory.h>
#include <fastrtps/types/DynamicDataHelper.hpp>
#include <fastrtps/types/DynamicPubSubType.h>
#include <fastrtps/types/DynamicTypePtr.h>
#include <fastrtps/types/TypeObjectFactory.h>
#include <google/protobuf/compiler/parser.h>
#include <google/protobuf/dynamic_message.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include "adf_lite_utile.h"
#include "ament_index_cpp/get_search_paths.hpp"
#include "cm/include/cm_config.h"
#include "google/protobuf/stubs/logging.h"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "idl/generated/cm_protobufTypeObject.h"
#include "idl/generated/cm_someipbufPubSubTypes.h"
#include "idl/generated/cm_someipbufTypeObject.h"
#include "idl/generated/proto_methodPubSubTypes.h"
#include "idl/generated/proto_methodTypeObject.h"
#include "proto_factory.h"
#include "data_tools_logger.hpp"
#include "sm/include/state_client.h"
#include "someip_deserialize_impl.h"
#include "proto/test/soc/dbg_msg.pb.h"

namespace hozon {

namespace netaos {
namespace topic {

EchoImpl::EchoImpl() {}

EchoImpl::~EchoImpl() {
    if (!_isStop) {
        Stop();
    }
}

void EchoImpl::Stop() {
    if (!_isStop) {
        _isStop = true;

        if (adflite_process_topics_map_.size() > 0) {
            //请求停止adf-lite
            hozon::netaos::data_tool_common::RequestCommand(adflite_process_topics_map_, "echo", false);
        }
    }
    else {

    }

    // SubBase::Stop();
}

void EchoImpl::Start(EchoOptions echo_options) {
    //判断是否有adf-lite topic
    if (0 < echo_options.topics.size()) {
        // 将topics按process进行分类
        for (auto& topic : echo_options.topics) {
            std::string process_name;
            int32_t res = hozon::netaos::data_tool_common::GetPartFromCmTopic(topic, 1, process_name);
            if (res == -1) {
                continue;
            }
            TOPIC_LOG_DEBUG << "topic :" << topic << " process_name:" << process_name;
            if (process_name.size() > 0) {
                adflite_process_topics_map_[process_name].push_back(topic);
            }
        }
        if (adflite_process_topics_map_.size() > 0) {
            //发送adf-lite request
            hozon::netaos::data_tool_common::RequestCommand(adflite_process_topics_map_, "echo", true);
        }
    }

    if (!echo_options.open_proto_log) {
        //close proto log
        google::protobuf::LogSilencer* logSilencer = new google::protobuf::LogSilencer();
        if (!logSilencer) {
            TOPIC_LOG_WARN << "creat protobuf LogSilencer failed";
        }
    }

    json_format_path_ = echo_options.json_format_path;

    _monitor_all = false;
    _method = false;
    SubBase::Start(echo_options.topics);

    while (!_isStop) {
        if (0 == topics_.size()) {
            std::cout << "waiting for topic ....." << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    SubBase::Stop();
    return;
}

void EchoImpl::OnDataAvailable(DataReader* reader) {

    SampleInfo info;
    std::string data_type = reader->get_topicdescription()->get_type_name();
    if ("CmProtoBuf" == data_type) {
        CmProtoBuf temp_cmProtoBuf;
        if (reader->take_next_sample(&(temp_cmProtoBuf), &info) == ReturnCode_t::RETCODE_OK) {
            if (info.valid_data) {
                std::string proto_name = temp_cmProtoBuf.name();
                google::protobuf::Message* prototype_msg = hozon::netaos::data_tool_common::ProtoFactory::getInstance()->GenerateMessageByType(proto_name);  // prototype_msg is immutable
                if (prototype_msg == NULL) {
                    std::cerr << "Cannot create prototype message from message descriptor";
                    return;
                }
                prototype_msg->ParseFromArray(temp_cmProtoBuf.str().data(), temp_cmProtoBuf.str().size());
                std::string debugstr;
                google::protobuf::TextFormat::PrintToString(*prototype_msg, &debugstr);
                std::cout << "----------new message---------" << std::endl;
                std::cout << debugstr << std::endl;

                // if (HasProtoField(*prototype_msg, "header")) {

                //     const google::protobuf::Message& header_msg = GetProtoReflectionMessage(*prototype_msg, "header");
                //     double publish_stamp = GetProtoReflectionDouble(header_msg, "publish_stamp");
                //     double data_stamp = GetProtoReflectionDouble(header_msg, "data_stamp");
                //     int32_t seq = GetProtoReflectionDouble(header_msg, "seq");
                //     double sensor_stamp = 0.0;

                //     const google::protobuf::Message& sensor_stamp_msg = GetProtoReflectionMessage(header_msg, "sensor_stamp");

                //     if (HasProtoField(sensor_stamp_msg, "lidar_stamp")) {
                //         sensor_stamp = GetProtoReflectionDouble(sensor_stamp_msg, "lidar_stamp");
                //     }
                //     else if (HasProtoField(sensor_stamp_msg, "radar_stamp")) {
                //         sensor_stamp = GetProtoReflectionDouble(sensor_stamp_msg, "radar_stamp");
                //     }
                //     else if (HasProtoField(sensor_stamp_msg, "uss_stamp")) {
                //         sensor_stamp = GetProtoReflectionDouble(sensor_stamp_msg, "uss_stamp");
                //     }
                //     else if (HasProtoField(sensor_stamp_msg, "chassis_stamp")) {
                //         sensor_stamp = GetProtoReflectionDouble(sensor_stamp_msg, "chassis_stamp");
                //     }
                //     else if (HasProtoField(sensor_stamp_msg, "camera_stamp")) {
                //         sensor_stamp = GetProtoReflectionDouble(sensor_stamp_msg, "camera_stamp");
                //     }
                //     else if (HasProtoField(sensor_stamp_msg, "imuins_stamp")) {
                //         sensor_stamp = GetProtoReflectionDouble(sensor_stamp_msg, "imuins_stamp");
                //     }
                //     else if (HasProtoField(sensor_stamp_msg, "gnss_stamp")) {
                //         sensor_stamp = GetProtoReflectionDouble(sensor_stamp_msg, "gnss_stamp");
                //     }
                //     else {

                //     }

                //     double recv_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
                //     std::cout << "latency. seq: " << seq << ", publish_stamp: " << std::fixed << std::setprecision(9) << (recv_time - publish_stamp) << ", recv_time: " << recv_time << std::endl;
                // }
            }
        }
    } 
    // else if ("CmSomeipBuf" == data_type) {
    //     CmSomeipBuf temp_cmProtoBuf;
    //     if (reader->take_next_sample(&(temp_cmProtoBuf), &info) == ReturnCode_t::RETCODE_OK) {
    //         if (info.valid_data) {
    //             std::string proto_name = temp_cmProtoBuf.name();
    //             google::protobuf::Message* prototype_msg = hozon::netaos::data_tool_common::ProtoFactory::getInstance()->GenerateMessageByType(proto_name);  // prototype_msg is immutable
    //             if (prototype_msg == NULL) {
    //                 std::cerr << "Cannot create prototype message from message descriptor";
    //                 return;
    //             }

    //             prototype_msg->ParseFromArray(temp_cmProtoBuf.str().data(), temp_cmProtoBuf.str().size());

    //             std::string debugstr;
    //             // google::protobuf::TextFormat::PrintToString(*prototype_msg, &debugstr);
    //             std::cout << "----------new message---------" << std::endl;
    //             std::cout << debugstr << std::endl;
    //         }
    //     }
    // } 
    else if ("CmSomeipBuf" == data_type) {
        CmSomeipBuf cmSomipBuf;
        if (reader->take_next_sample(&(cmSomipBuf), &info) == ReturnCode_t::RETCODE_OK) {
            if (info.valid_data) {
                std::string topic_name = cmSomipBuf.name();
                std::string debugstr = hozon::netaos::someip_deserialize::SomeipDeserializeImpl::getInstance(json_format_path_)->deserialize(cmSomipBuf.str().data(), cmSomipBuf.str().size(), topic_name);
                std::cout << "----------new someip message start---------" << std::endl;
                std::cout << debugstr << std::endl;
                std::cout << "----------new someip message end---------" << std::endl;
            }
        }
    } else {
        ProtoMethodBase protoMethod_data;
        if (reader->take_next_sample(&(protoMethod_data), &info) == ReturnCode_t::RETCODE_OK) {
            if (info.valid_data) {
                std::string proto_name = protoMethod_data.name();
                google::protobuf::Message* prototype_msg = hozon::netaos::data_tool_common::ProtoFactory::getInstance()->GenerateMessageByType(proto_name);  // prototype_msg is immutable
                if (prototype_msg == NULL) {
                    std::cerr << "Cannot create prototype message from message descriptor";
                    return;
                }
                prototype_msg->ParseFromArray(protoMethod_data.str().data(), protoMethod_data.str().size());
                std::string debugstr;
                google::protobuf::TextFormat::PrintToString(*prototype_msg, &debugstr);
                std::cout << "----------new message---------" << std::endl;
                std::cout << debugstr << std::endl;
            }
        }
    }
}

}  // namespace topic
}  //namespace netaos
}  //namespace hozon