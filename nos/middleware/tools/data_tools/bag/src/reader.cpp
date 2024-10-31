#include "reader.h"
#include "codec/include/codec_def.h"
#include "codec/include/decoder.h"
#include "codec/include/decoder_factory.h"
#include "data_tools_logger.hpp"
#include "google/protobuf/util/json_util.h"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "idl/generated/cm_protobufTypeObject.h"
#include "idl/generated/cm_someipbufPubSubTypes.h"
#include "idl/generated/cm_someipbufTypeObject.h"
#include "idl/generated/proto_methodPubSubTypes.h"
#include "idl/generated/proto_methodTypeObject.h"
#include "impl/reader_impl.h"
#include "message_process.h"
#include "proto_factory.h"
#include "proto_utility.h"

using hozon::netaos::codec::Decoder;
using hozon::netaos::codec::DecoderFactory;

namespace hozon {
namespace netaos {
namespace bag {

using namespace eprosima::fastrtps::rtps;

Reader::Reader() {
    reader_impl_ = std::make_unique<ReaderImpl>();
};

Reader::~Reader() {
    if (reader_impl_) {
        reader_impl_ = nullptr;
    }
};

ReaderErrorCode Reader::Open(const std::string uri, const std::string storage_id) {
    return reader_impl_->Open(uri, storage_id);
};

void Reader::Close() {
    reader_impl_->Close();
};

bool Reader::HasNext() {
    return reader_impl_->HasNext();
};

TopicMessage Reader::ReadNext() {
    std::shared_ptr<rosbag2_storage::SerializedBagMessage> message_ptr = reader_impl_->ReadNext();
    TopicMessage message_vec;
    message_vec.topic = message_ptr->topic_name;
    message_vec.type = reader_impl_->GetAllTopicsAndTypes()[message_ptr->topic_name];
    message_vec.time = message_ptr->time_stamp;
    message_vec.data.resize(message_ptr->serialized_data->buffer_length);
    std::memcpy(message_vec.data.data(), message_ptr->serialized_data->buffer, message_ptr->serialized_data->buffer_length);
    return message_vec;
};

ReaderErrorCode Reader::ReadNextAsJson(double& time, std::string& topic_name, std::string& json_data, BinaryType& binary_type, std::vector<uint8_t>& binary_data) {
    binary_type = BinaryType::kBinaryType_Null;
    TopicMessage message = ReadNext();
    if ("" == message.topic) {
        BAG_LOG_ERROR << "can not get the data type of topic: " + message.topic + ".";
        return ReaderErrorCode::TAKE_NEXT_MESSAGE_FAILED;
    }
    topic_name = message.topic;
    double public_time = 0;
    double sensor_time = 0;

    std::string proto_name;
    std::vector<char> proto_message;
    if ("CmProtoBuf" == message.type) {
        std::shared_ptr<SerializedPayload_t> payload_ptr = Convert2Payload(message);
        CmProtoBuf cm_buf;
        CmProtoBufPubSubType cm_pub_sub;
        cm_pub_sub.deserialize(payload_ptr.get(), &cm_buf);
        proto_name = cm_buf.name();
        proto_message = cm_buf.str();
    } else if ("ProtoMethodBase" == message.type) {
        std::shared_ptr<SerializedPayload_t> payload_ptr = Convert2Payload(message);
        ProtoMethodBase method_buf;
        ProtoMethodBasePubSubType method_pub_sub;
        method_pub_sub.deserialize(payload_ptr.get(), &method_buf);
        proto_name = method_buf.name();
        proto_message = method_buf.str();
    } else if (("CmSomeipBuf" == message.type)) {
        std::shared_ptr<SerializedPayload_t> payload_ptr = Convert2Payload(message);
        CmSomeipBuf cm_buf;
        CmSomeipBufPubSubType cm_pub_sub;
        cm_pub_sub.deserialize(payload_ptr.get(), &cm_buf);
        proto_name = cm_buf.name();
        proto_message = cm_buf.str();
    } else {
        message.type;
        return ReaderErrorCode::FAILED_IDL_TYPE_SUPPORT;
    }

    std::shared_ptr<google::protobuf::Message> proto_data;
    if (proto_message.size() > 0) {
        //process jpg、pcd
        bool is_pcd = false;
        if (std::find(reader_impl_->pcd_topic_list_.begin(), reader_impl_->pcd_topic_list_.end(), message.topic) != reader_impl_->pcd_topic_list_.end()) {
            //pcd
            hozon::soc::PointCloud* msg = new hozon::soc::PointCloud;
            if ("/soc/rawpointcloud" == topic_name) {
                topic_name = "/soc/pointcloud";
                if (reader_impl_->is_first_handle_raw_point) {
                    std::cout << "\033[33m"
                              << "wallning: topic: '/soc/rawpointcloud' will return as '/soc/pointcloud', Please ensure that this program does not read '/soc/pointcloud'"
                              << "\033[0m" << std::endl;
                    reader_impl_->is_first_handle_raw_point = false;
                    /* code */
                }
                hozon::soc::RawPointCloud raw_point;
                raw_point.ParseFromArray(proto_message.data(), proto_message.size());
                if (0 != MessageProcess::Instance("./").Parse(raw_point, *msg)) {
                    BAG_LOG_WARN << "Process pointcloud fail!";
                }
            } else {
                msg->ParseFromArray(proto_message.data(), proto_message.size());
            }
            reader_impl_->ToPcd(*msg, binary_data);
            msg->clear_points();
            public_time = msg->mutable_header()->publish_stamp();
            sensor_time = msg->mutable_header()->sensor_stamp().lidar_stamp();
            proto_data.reset(msg);
            binary_type = BinaryType::kBinaryType_Pcd;
            is_pcd = true;
        } else if (std::find(reader_impl_->jpg_topic_list_.begin(), reader_impl_->jpg_topic_list_.end(), message.topic) != reader_impl_->jpg_topic_list_.end()) {
            //jpg
            hozon::soc::CompressedImage* msg = new hozon::soc::CompressedImage;
            msg->ParseFromArray(proto_message.data(), proto_message.size());
            reader_impl_->ToJpg(message.topic, *msg, binary_data);
            msg->set_data(std::string());
            proto_data.reset(msg);
            public_time = msg->mutable_header()->publish_stamp();
            sensor_time = msg->mutable_header()->sensor_stamp().camera_stamp();
            binary_type = BinaryType::kBinaryType_Jpg;
        } else {
            proto_data = std::shared_ptr<google::protobuf::Message>(hozon::netaos::data_tool_common::ProtoFactory::getInstance()->GenerateMessageByType(proto_name));
            if (nullptr == proto_data) {
                BAG_LOG_ERROR << "get proto:" << proto_name << " type faild";
                return ReaderErrorCode::FILE_FAILED;
            }
            proto_data->ParseFromArray(proto_message.data(), proto_message.size());
            //get time
            if (hozon::netaos::data_tool_common::ProtoUtility::HasProtoField(*(proto_data.get()), "header")) {
                const google::protobuf::Message& header_msg = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionMessage(*(proto_data.get()), "header");
                public_time = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionDouble(header_msg, "publish_stamp");
                const google::protobuf::Message& sensor_stamp_msg = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionMessage(header_msg, "sensor_stamp");
                if (hozon::netaos::data_tool_common::ProtoUtility::HasProtoField(sensor_stamp_msg, "uss_stamp")) {
                    sensor_time = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionDouble(sensor_stamp_msg, "uss_stamp");
                } else if (hozon::netaos::data_tool_common::ProtoUtility::HasProtoField(sensor_stamp_msg, "chassis_stamp")) {
                    sensor_time = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionDouble(sensor_stamp_msg, "chassis_stamp");
                } else if (hozon::netaos::data_tool_common::ProtoUtility::HasProtoField(sensor_stamp_msg, "imuins_stamp")) {
                    sensor_time = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionDouble(sensor_stamp_msg, "imuins_stamp");
                } else if (hozon::netaos::data_tool_common::ProtoUtility::HasProtoField(sensor_stamp_msg, "gnss_stamp")) {
                    sensor_time = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionDouble(sensor_stamp_msg, "gnss_stamp");
                } else if (hozon::netaos::data_tool_common::ProtoUtility::HasProtoField(sensor_stamp_msg, "radar_stamp")) {
                    sensor_time = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionDouble(sensor_stamp_msg, "radar_stamp");
                } else if (hozon::netaos::data_tool_common::ProtoUtility::HasProtoField(sensor_stamp_msg, "camera_stamp")) {
                    sensor_time = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionDouble(sensor_stamp_msg, "camera_stamp");
                } else if (hozon::netaos::data_tool_common::ProtoUtility::HasProtoField(sensor_stamp_msg, "lidar_stamp")) {
                    sensor_time = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionDouble(sensor_stamp_msg, "lidar_stamp");
                }
            }
        }
        // std::cout << "***recv_time=" << std::fixed << std::setprecision(9) << message.time / 1000000000.0 << "; public_time=" << std::fixed << std::setprecision(9) << public_time
        //           << "; sensor_time=" << std::fixed << std::setprecision(9) << sensor_time << std::endl;
        if (sensor_time != 0 && sensor_time < public_time) {
            time = sensor_time;
        } else if (public_time != 0 && public_time < message.time / 1000000000.0) {
            time = public_time;
        } else {
            time = message.time / 1000000000.0;
        }
        //process json
        google::protobuf::util::JsonPrintOptions json_print_options;
        google::protobuf::util::Status status = google::protobuf::util::MessageToJsonString(*proto_data, &json_data, json_print_options);
        if (!status.ok()) {
            BAG_LOG_ERROR << status.message().ToString();
            return ReaderErrorCode::TO_JSON_FAILED;
        }
        return ReaderErrorCode::SUCCESS;
    }
    return ReaderErrorCode::SUCCESS;
}

std::shared_ptr<SerializedPayload_t> Reader::Convert2Payload(const TopicMessage& message) {
    std::shared_ptr<SerializedPayload_t> payload_ptr = std::make_shared<SerializedPayload_t>();
    payload_ptr->reserve(message.data.size());
    memcpy(payload_ptr->data, message.data.data(), message.data.size());
    payload_ptr->length = message.data.size();
    return payload_ptr;
};

std::vector<char> Reader::GetProtoMessage(const TopicMessage& message, const std::string proto_name) {
    std::vector<char> proto_message;
    if ("CmProtoBuf" == message.type) {
        std::shared_ptr<SerializedPayload_t> payload_ptr = Convert2Payload(message);
        CmProtoBuf cm_buf;
        CmProtoBufPubSubType cm_pub_sub;
        cm_pub_sub.deserialize(payload_ptr.get(), &cm_buf);
        if (proto_name == cm_buf.name()) {
            proto_message = cm_buf.str();
        }

    } else if ("ProtoMethodBase" == message.type) {
        std::shared_ptr<SerializedPayload_t> payload_ptr = Convert2Payload(message);
        ProtoMethodBase method_buf;
        ProtoMethodBasePubSubType method_pub_sub;
        method_pub_sub.deserialize(payload_ptr.get(), &method_buf);
        if (proto_name == method_buf.name()) {
            proto_message = method_buf.str();
        }
    } else if ("CmSomeipBuf" == message.type) {
        std::shared_ptr<SerializedPayload_t> payload_ptr = Convert2Payload(message);
        CmSomeipBuf cm_buf;
        CmSomeipBufPubSubType cm_pub_sub;
        cm_pub_sub.deserialize(payload_ptr.get(), &cm_buf);
        if (proto_name == cm_buf.name()) {
            proto_message = cm_buf.str();
        }
    }
    return proto_message;
};

std::map<std::string, std::string> Reader::GetAllTopicsAndTypes() const {
    return reader_impl_->GetAllTopicsAndTypes();
};

void Reader::SetFilter(const std::vector<std::string>& topics) {
    reader_impl_->SetFilter(topics);
};

void Reader::ResetFilter() {
    reader_impl_->ResetFilter();
};

void Reader::Seek(const int64_t& timestamp) {
    reader_impl_->Seek(timestamp);
};

rosbag2_storage::BagMetadata Reader::GetMetadata() {
    return reader_impl_->GetMetadata();
};

}  // namespace bag
}  //namespace netaos
}  //namespace hozon
