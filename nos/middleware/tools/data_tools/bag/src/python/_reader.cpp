#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <fastdds/rtps/common/SerializedPayload.h>

#include "./pybind11.hpp"

namespace hozon {
namespace netaos {
namespace bag_py {

struct TopicMessage {
    std::string topic;
    std::string type;
    int64_t time;
    std::vector<char> data;
};

enum ReaderErrorCode { READ_FILE_FAILED = -1, SUCCESS = 0 };

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
   * Ask whether the underlying bagfile contains at least one more message.
   *
   * \return true if storage contains at least one more message
   * \throws runtime_error if the Reader is not open.
   */
    bool HasNext();

    /**
   * Read next message from storage. Will throw if no more messages are available.
   *
   * Expected usage:
   * if (writer.has_next()) message = writer.read_next();
   *
   * \return next message in serialized form
   * \throws runtime_error if the Reader is not open.
   */
    TopicMessage ReadNext();

    /**
   *  Deserialize message to secific ptoto form, Will throw if no more messages are available.
   *
   * Expected usage:
   * if (writer.has_next()) message = writer.read_next();
   * \return next message in non-serialized form
   * \throws runtime_error if the Reader is not open.
   */
    template <typename ProtoType>
    ProtoType DeserializeToProto(TopicMessage& topic_message) {
        ProtoType proto_data;
        std::vector<char> proto_message = GetProtoMessage(topic_message, proto_data.GetTypeName());
        if (proto_message.size() > 0) {
            proto_data.ParseFromArray(proto_message.data(), proto_message.size());
        }
        return proto_data;
    }

    /**
   * Read next message from storage and deserialized is to  "CmProtoBuf" or "ProtoMethodBase"  form, Will throw if no more messages are available.
   *
   * Expected usage:
   * if (writer.has_next()) message = writer.read_next();
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
   * Read next message from storage and deserialized is to secific ptoto form, Will throw if no more messages are available.
   *
   * Expected usage:
   * if (writer.has_next()) message = writer.read_next();
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

   private:
    std::shared_ptr<eprosima::fastrtps::rtps::SerializedPayload_t> Convert2Payload(const TopicMessage& message);
    std::vector<char> GetProtoMessage(const TopicMessage& message, std::string proto_name);
    std::unique_ptr<ReaderImpl> reader_impl_;
};

}  // namespace bag_py
}  //namespace netaos
}  //namespace hozon
