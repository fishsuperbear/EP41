#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <fastdds/rtps/common/SerializedPayload.h>
#include "rosbag2_storage/bag_metadata.hpp"

namespace hozon {
namespace netaos {
namespace bag {

struct TopicMessage {
    std::string topic;       //topic名称
    std::string type;        //数据类型（cm上的idl类型）
    int64_t time;            //时间戳（纳秒）
    std::vector<char> data;  //序列化数据
};

enum class ReaderErrorCode {
    SUCCESS = 0,
    FILE_FAILED = 1,
    READ_FILE_FAILED = 2,          //打开文件失败
    INVALID_FRAME = 3,             //无效帧
    TAKE_NEXT_MESSAGE_FAILED = 4,  //读取下一帧数据失败
    TO_JSON_FAILED = 5,            //proto转json失败
    FAILED_IDL_TYPE_SUPPORT = 6    //数据类型不支持
};

enum BinaryType : uint32_t { kBinaryType_Null = 0, kBinaryType_Jpg = 1, kBinaryType_Pcd = 2 };

class ReaderImpl;

class Reader final {
   public:
    explicit Reader();

    ~Reader();

    /**
   * Throws if file could not be opened.
   * This must be called before any other function is used.
   * The rosbag is automatically closed on destruction.
   *
   * \param uri bag path
   * \param storage_id bag format
   */
    ReaderErrorCode Open(const std::string uri, const std::string storage_id = "mcap");

    /**
   * Closing the reader instance.
   */
    void Close();

    /**
   * Ask bag file for all topics (including their type) that were recorded.
   *
   * \return map of <topic_name, type_name>
   * \throws runtime_error if the Reader is not open.
   */
    std::map<std::string, std::string> GetAllTopicsAndTypes() const;

    /**
   * Set filters to adhere to during reading.
   *
   * \param storage_filter Filter to apply to reading
   * \throws runtime_error if the Reader is not open.
   */
    void SetFilter(const std::vector<std::string>& topics);

    /**
   * Reset all filters for reading.
   */
    void ResetFilter();

    /**
   * Skip to a specific timestamp for reading.
   */
    void Seek(const int64_t& timestamp);

    /**
   * Ask whether the underlying bagfile contains at least one more message.
   *
   * \return true if storage contains at least one more message
   * \throws runtime_error if the Reader is not open.
   */
    bool HasNext();

    /**
   * Read next message from storage and deserialized is to secific ptoto form, Will throw if no more messages are available.
   *
   * Expected usage:
   * if (reader.has_next()) message = reader.ReadNextProtoMessage();
   * \return next message in non-serialized form
   * \throws runtime_error if the Reader is not open.
   */
    template <typename ProtoType>
    ProtoType ReadNextProtoMessage() {
        ProtoType proto_data;
        TopicMessage message = ReadNext();
        if ("" == message.topic) {
            throw std::runtime_error("can not get the data type of topic: " + message.topic + ".");
        }
        std::vector<char> proto_message = GetProtoMessage(message, proto_data.GetTypeName());
        if (proto_message.size() > 0) {
            proto_data.ParseFromArray(proto_message.data(), proto_message.size());
        }
        return proto_data;
    };

    /**
   * Read next message from storage and deserialized is to  "CmProtoBuf" or "ProtoMethodBase"  form, Will throw if no more messages are available.
   *
   * Expected usage:
   * if (reader.has_next()) message = reader.ReadNextIdlMessage();
   * \return next message in non-serialized form
   * \throws runtime_error if the Reader is not open.
   */
    template <typename MessagePubSubType, typename MessageType>
    MessageType ReadNextIdlMessage() {
        MessageType msg;
        MessagePubSubType msg_sub;
        TopicMessage message = ReadNext();
        if (msg_sub.getName() == message.type) {
            std::shared_ptr<eprosima::fastrtps::rtps::SerializedPayload_t> paylod_ptr = Convert2Payload(message);
            msg_sub.deserialize(paylod_ptr.get(), &msg);
        }
        return msg;
    };

    /**
   * Read next message from storage and save as json,
   * if message is point cloud data will return pcd format data too,
   * and if message is camera data will return jpg format data too,
   *  Will throw if no more messages are available.
   *
   * \param json_data next message as json
   * \param binary_type the type of the return data
   * \param binary_data the binary data in pcd or jpg
   * Expected usage:
   * if (reader.has_next()) message = reader.ReadNextAsjson();
   *
   * \return ReaderErrorCode
   * \throws runtime_error if the Reader is not open.
   */
    ReaderErrorCode ReadNextAsJson(double& time, std::string& topic_name, std::string& json_data, BinaryType& binary_type, std::vector<uint8_t>& binary_data);

    /**
   * Read next message from storage. Will throw if no more messages are available.
   *
   * Expected usage:
   * if (reader.has_next()) message = reader.ReadNext();
   *
   * \return next message in serialized form
   * \throws runtime_error if the Reader is not open.
   */
    TopicMessage ReadNext();

    /**
   *  Deserialize message to secific ptoto form, Will throw if no more messages are available.
   *
   * \return next message in non-serialized form
   * \throws runtime_error if the Reader is not open.
   */
    template <typename ProtoType>
    ProtoType DeserializeToProto(const TopicMessage& topic_message) {
        ProtoType proto_data;
        std::vector<char> proto_message = GetProtoMessage(topic_message, proto_data.GetTypeName());
        if (proto_message.size() > 0) {
            proto_data.ParseFromArray(proto_message.data(), proto_message.size());
        }

        return proto_data;
    }

    /**
   * Deserialize message to secific idl form, Will throw if no more messages are available.
   *
   * \return next message in idl form
   * \throws runtime_error if the Reader is not open.
   */
    template <typename MessagePubSubType, typename MessageType>
    MessageType DeserializeToIdl(const TopicMessage& message) {
        MessageType msg;
        MessagePubSubType msg_sub;
        if (msg_sub.getName() == message.type) {
            std::shared_ptr<eprosima::fastrtps::rtps::SerializedPayload_t> paylod_ptr = Convert2Payload(message);
            msg_sub.deserialize(paylod_ptr.get(), &msg);
        }
        return msg;
    };

    rosbag2_storage::BagMetadata GetMetadata();

   private:
    std::shared_ptr<eprosima::fastrtps::rtps::SerializedPayload_t> Convert2Payload(const TopicMessage& message);
    std::vector<char> GetProtoMessage(const TopicMessage& message, const std::string proto_name);
    std::unique_ptr<ReaderImpl> reader_impl_;
};

}  // namespace bag
}  //namespace netaos
}  //namespace hozon
