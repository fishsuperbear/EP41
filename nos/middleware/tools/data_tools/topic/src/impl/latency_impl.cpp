
#include "impl/latency_impl.h"
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
#include "data_tools_logger.hpp"
#include "google/protobuf/stubs/logging.h"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "idl/generated/cm_protobufTypeObject.h"
#include "idl/generated/proto_methodPubSubTypes.h"
#include "idl/generated/proto_methodTypeObject.h"
#include "idl/generated/zerocopy_imagePubSubTypes.h"
#include "idl/generated/zerocopy_imageTypeObject.h"
#include "proto_factory.h"
#include "proto_utility.h"
#include "sm/include/state_client.h"
#include "proto/soc/point_cloud.pb.h"
#include "proto/soc/sensor_image.pb.h"
#include "proto/test/soc/dbg_msg.pb.h"

namespace hozon {

namespace netaos {
namespace topic {

LatencyImpl::LatencyImpl() {}

LatencyImpl::~LatencyImpl() {
    if (!_isStop) {
        Stop();
    }
}

void LatencyImpl::Stop() {
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

void LatencyImpl::Start(LatencyOptions latency_options) {
    latency_options_ = latency_options;

    TOPIC_LOG_DEBUG << "topic latency start.";

    //判断是否有adf-lite topic
    if (0 < latency_options.topics.size()) {
        // 将topics按process进行分类
        for (auto& topic : latency_options.topics) {
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

    _monitor_all = false;
    _method = false;
    SubBase::Start(latency_options.topics);

    while (!_isStop) {
        if (0 == topics_.size()) {
            std::cout << "waiting for topic ....." << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    SubBase::Stop();
    return;
}

void LatencyImpl::OnDataAvailable(DataReader* reader) {
    // BAG_LOG_DEBUG << "on_data_available!";
    SampleInfo info;
    std::string data_type = reader->get_topicdescription()->get_type_name();
    std::string topic = reader->get_topicdescription()->get_name();
    if ("CmProtoBuf" == data_type) {
        CmProtoBuf temp_cmProtoBuf;

        if (reader->take_next_sample(&(temp_cmProtoBuf), &info) == ReturnCode_t::RETCODE_OK) {
            if (info.valid_data) {
                std::string proto_name = temp_cmProtoBuf.name();
                std::shared_ptr<google::protobuf::Message> message;
                if (proto_name.find("PointCloud") != std::string::npos) {
                    message = std::shared_ptr<google::protobuf::Message>(new hozon::soc::PointCloud());
                } else if (proto_name.find("CompressedImage") != std::string::npos) {
                    message = std::shared_ptr<google::protobuf::Message>(new hozon::soc::CompressedImage());
                } else if (proto_name.find("Image") != std::string::npos) {
                    message = std::shared_ptr<google::protobuf::Message>(new hozon::soc::Image());
                } else {
                    message = std::shared_ptr<google::protobuf::Message>(hozon::netaos::data_tool_common::ProtoFactory::getInstance()->GenerateMessageByType(proto_name));
                }

                if (message == NULL) {
                    std::cerr << "Cannot create prototype message from message descriptor";
                    return;
                }
                bool parse_res = message->ParseFromArray(temp_cmProtoBuf.str().data(), temp_cmProtoBuf.str().size());

                double recv_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

                if (hozon::netaos::data_tool_common::ProtoUtility::HasProtoField(*message, "header")) {

                    const google::protobuf::Message& header_msg = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionMessage(*message, "header");
                    double publish_stamp = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionDouble(header_msg, "publish_stamp");
                    double data_stamp = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionDouble(header_msg, "data_stamp");
                    int32_t seq = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionInt32(header_msg, "seq");
                    double sensor_stamp = 0.0;

                    const google::protobuf::Message& sensor_stamp_msg = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionMessage(header_msg, "sensor_stamp");

                    if (hozon::netaos::data_tool_common::ProtoUtility::HasProtoField(sensor_stamp_msg, "lidar_stamp")) {
                        sensor_stamp = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionDouble(sensor_stamp_msg, "lidar_stamp");
                    } else if (hozon::netaos::data_tool_common::ProtoUtility::HasProtoField(sensor_stamp_msg, "radar_stamp")) {
                        sensor_stamp = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionDouble(sensor_stamp_msg, "radar_stamp");
                    } else if (hozon::netaos::data_tool_common::ProtoUtility::HasProtoField(sensor_stamp_msg, "uss_stamp")) {
                        sensor_stamp = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionDouble(sensor_stamp_msg, "uss_stamp");
                    } else if (hozon::netaos::data_tool_common::ProtoUtility::HasProtoField(sensor_stamp_msg, "chassis_stamp")) {
                        sensor_stamp = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionDouble(sensor_stamp_msg, "chassis_stamp");
                    } else if (hozon::netaos::data_tool_common::ProtoUtility::HasProtoField(sensor_stamp_msg, "camera_stamp")) {
                        sensor_stamp = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionDouble(sensor_stamp_msg, "camera_stamp");
                    } else if (hozon::netaos::data_tool_common::ProtoUtility::HasProtoField(sensor_stamp_msg, "imuins_stamp")) {
                        sensor_stamp = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionDouble(sensor_stamp_msg, "imuins_stamp");
                    } else if (hozon::netaos::data_tool_common::ProtoUtility::HasProtoField(sensor_stamp_msg, "gnss_stamp")) {
                        sensor_stamp = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionDouble(sensor_stamp_msg, "gnss_stamp");
                    } else {
                    }

                    std::cout << "topic: " << topic << ", seq: " << seq << ", latency: " << std::fixed << std::setprecision(9) << (recv_time - publish_stamp) << ", publish_stamp: " << publish_stamp
                              << ", recv_time: " << recv_time;
                    if (sensor_stamp > 0.0) {
                        std::cout << ", sensor_stamp: " << sensor_stamp;
                    }
                    if (data_stamp > 0.0) {
                        std::cout << ", data_stamp: " << data_stamp;
                    }
                    std::cout << std::endl;
                }
            }
        }
    } else if ("ProtoMethodBase" == data_type) {
        ProtoMethodBase protoMethod_data;
        if (reader->take_next_sample(&(protoMethod_data), &info) == ReturnCode_t::RETCODE_OK) {
            if (info.valid_data) {
                std::string proto_name = protoMethod_data.name();
                std::shared_ptr<google::protobuf::Message> message(hozon::netaos::data_tool_common::ProtoFactory::getInstance()->GenerateMessageByType(proto_name));
                if (message == NULL) {
                    std::cerr << "Cannot create prototype message from message descriptor";
                    return;
                }
                message->ParseFromArray(protoMethod_data.str().data(), protoMethod_data.str().size());
                std::string debugstr;
                google::protobuf::TextFormat::PrintToString(*message, &debugstr);
                std::cout << debugstr << std::endl;
            }
        }
    } else if (data_type.find("ZeroCopyImg8M420") != std::string::npos) {
        // ZeroCopyImg8M420 temp_zerocopy;
        // ReturnCode_t ret = reader->take_next_sample(&(temp_zerocopy), &info);
        // if (ret == ReturnCode_t::RETCODE_OK) {
        // }

        FASTDDS_SEQUENCE(ZeroCopyImg8M420Seq, ZeroCopyImg8M420);
        ZeroCopyImg8M420Seq data;
        SampleInfoSeq infos;

        if (ReturnCode_t::RETCODE_OK == reader->take(data, infos)) {
            // Iterate over each LoanableCollection in the SampleInfo sequence
            for (LoanableCollection::size_type i = 0; i < infos.length(); ++i) {
                // Check whether the DataSample contains data or is only used to communicate of a
                // change in the instance
                if (infos[i].valid_data) {
                    // Print the data.
                    const ZeroCopyImg8M420& sample = data[i];
                    double recv_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
                    std::cout << "topic: " << topic << ", seq: " << sample.frame_count() << ", latency: " << std::fixed << std::setprecision(9)
                              << (recv_time - sample.pushlish_timestamp() / 1000000000.0) << ", publish_stamp: " << sample.pushlish_timestamp() / 1000000000.0 << ", recv_time: " << recv_time
                              << ", sensor_stamp: " << sample.sensor_timestamp() / 1000000000.0 << std::endl;
                }
            }

            reader->return_loan(data, infos);
        }
    } else if (data_type.find("ZeroCopyImg2M422") != std::string::npos) {
        // ZeroCopyImg2M422 temp_zerocopy;
        // ReturnCode_t ret = reader->take_next_sample(&(temp_zerocopy), &info);
        // if (ret == ReturnCode_t::RETCODE_OK) {
        // }

        FASTDDS_SEQUENCE(ZeroCopyImg2M422Seq, ZeroCopyImg2M422);
        ZeroCopyImg2M422Seq data;
        SampleInfoSeq infos;

        if (ReturnCode_t::RETCODE_OK == reader->take(data, infos)) {
            // Iterate over each LoanableCollection in the SampleInfo sequence
            for (LoanableCollection::size_type i = 0; i < infos.length(); ++i) {
                // Check whether the DataSample contains data or is only used to communicate of a
                // change in the instance
                if (infos[i].valid_data) {
                    // Print the data.
                    const ZeroCopyImg2M422& sample = data[i];
                    double recv_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
                    std::cout << "topic: " << topic << ", seq: " << sample.frame_count() << ", latency: " << std::fixed << std::setprecision(9)
                              << (recv_time - sample.pushlish_timestamp() / 1000000000.0) << ", publish_stamp: " << sample.pushlish_timestamp() / 1000000000.0 << ", recv_time: " << recv_time
                              << ", sensor_stamp: " << sample.sensor_timestamp() / 1000000000.0 << std::endl;
                }
            }

            reader->return_loan(data, infos);
        }
    } else if (data_type.find("ZeroCopyImg3M422") != std::string::npos) {
        // ZeroCopyImg3M422 temp_zerocopy;
        // ReturnCode_t ret = reader->take_next_sample(&(temp_zerocopy), &info);
        // if (ret == ReturnCode_t::RETCODE_OK) {
        // }

        FASTDDS_SEQUENCE(ZeroCopyImg3M422Seq, ZeroCopyImg3M422);
        ZeroCopyImg3M422Seq data;
        SampleInfoSeq infos;

        if (ReturnCode_t::RETCODE_OK == reader->take(data, infos)) {
            // Iterate over each LoanableCollection in the SampleInfo sequence
            for (LoanableCollection::size_type i = 0; i < infos.length(); ++i) {
                // Check whether the DataSample contains data or is only used to communicate of a
                // change in the instance
                if (infos[i].valid_data) {
                    // Print the data.
                    const ZeroCopyImg3M422& sample = data[i];
                    double recv_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
                    std::cout << "topic: " << topic << ", seq: " << sample.frame_count() << ", latency: " << std::fixed << std::setprecision(9)
                              << (recv_time - sample.pushlish_timestamp() / 1000000000.0) << ", publish_stamp: " << sample.pushlish_timestamp() / 1000000000.0 << ", recv_time: " << recv_time
                              << ", sensor_stamp: " << sample.sensor_timestamp() / 1000000000.0 << std::endl;
                }
            }

            reader->return_loan(data, infos);
        }
    } else {
    }
}

}  // namespace topic
}  //namespace netaos
}  //namespace hozon