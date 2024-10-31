

#include "fastdds/rtps/common/SerializedPayload.h"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "idl/generated/cm_protobufTypeObject.h"
#include "idl/generated/proto_methodPubSubTypes.h"
#include "idl/generated/proto_methodTypeObject.h"
#include "proto_factory.h"
#include "data_tools_logger.hpp"
#include "rosbag2_storage/ros_helper.hpp"
#include "rosbag2_storage/storage_interfaces/read_write_interface.hpp"
#include "rosbag2_storage_record/reord_proto_factory.h"
#include "cyber/record/record_reader.h"
#include "cyber/record/record_writer.h"

using namespace hozon::netaos::data_tool_common;

namespace rosbag2_storage_plugins {

using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;
static const char RECORD_FILE_EXTENSION[] = ".record";

// static const char RECORD__LOG_NAME[] = " [rosbag2_storage_record] ";

/**
 * A storage implementation for the MCAP file format.
 */
class RecordStorage : public rosbag2_storage::storage_interfaces::ReadWriteInterface {
   public:
    RecordStorage();
    ~RecordStorage() override;

    /** BaseIOInterface **/
    void open(const rosbag2_storage::StorageOptions& storage_options, rosbag2_storage::storage_interfaces::IOFlag io_flag = rosbag2_storage::storage_interfaces::IOFlag::READ_WRITE) override;
    void open(const std::string& uri, rosbag2_storage::storage_interfaces::IOFlag io_flag = rosbag2_storage::storage_interfaces::IOFlag::READ_WRITE);
    void add_attach(std::shared_ptr<const rosbag2_storage::Attachment> attachment) override;
    /** BaseInfoInterface **/
    rosbag2_storage::BagMetadata get_metadata() override;
    std::string get_relative_file_path() const override;
    uint64_t get_bagfile_size() const override;
    std::string get_storage_identifier() const override;

    /** BaseReadInterface **/
    bool has_next() override;
    std::shared_ptr<rosbag2_storage::SerializedBagMessage> read_next() override;
    std::vector<rosbag2_storage::TopicMetadata> get_all_topics_and_types() override;
    std::shared_ptr<rosbag2_storage::Attachment> read_Attachment(std::string name) override;
    void get_all_attachments_filepath(std::vector<std::string> &attach_list) override;

    /** ReadOnlyInterface **/
    void set_filter(const rosbag2_storage::StorageFilter& storage_filter) override;
    void reset_filter() override;
    void seek(const rcutils_time_point_value_t& time_stamp) override;

    /** ReadWriteInterface **/
    uint64_t get_minimum_split_file_size() const override;

    /** BaseWriteInterface **/
    void write(std::shared_ptr<const rosbag2_storage::SerializedBagMessage> msg) override;
    void write(const std::vector<std::shared_ptr<const rosbag2_storage::SerializedBagMessage>>& msg) override;
    void create_topic(const rosbag2_storage::TopicMetadata& topic) override;
    void remove_topic(const rosbag2_storage::TopicMetadata& topic) override;

    // void update_metadata(const rosbag2_storage::BagMetadata&) override;
    void RegisterSplitFileCallback(std::function<void(std::string, uint64_t, bool)> callback_func) override;

   private:
    void open_impl(const std::string& uri, rosbag2_storage::storage_interfaces::IOFlag io_flag, const std::string& storage_config_uri);
    bool read_and_enqueue_message();
    bool message_indexes_present();
    void ensure_summary_read();
    void on_file_event(std::string absolute_path, uint64_t na_time, bool is_open);

    uint64_t begin_time_ = 0;
    std::unique_ptr<apollo::cyber::record::RecordReader> record_reader_;
    std::unique_ptr<apollo::cyber::record::RecordWriter> record_writer_;
    std::optional<rosbag2_storage::storage_interfaces::IOFlag> opened_as_;
    rosbag2_storage::StorageOptions storage_options_;

    std::map<std::string, std::string> regist_proto_type_;

    std::string relative_path_;
    std::shared_ptr<rosbag2_storage::SerializedBagMessage> next_;
    rosbag2_storage::BagMetadata metadata_{};
    // std::unordered_map<std::string, rosbag2_storage::TopicInformation> topics_;
    rosbag2_storage::StorageFilter storage_filter_{};
    std::map<std::string, std::string> topics_;
    std::vector<std::function<void(std::string, uint64_t, bool)>> split_file_callback_;
};

RecordStorage::RecordStorage() {
    metadata_.storage_identifier = get_storage_identifier();
    metadata_.message_count = 0;
}

RecordStorage::~RecordStorage() {
    if (record_writer_) {
        record_writer_->Close();
    }
}

/** BaseIOInterface **/
void RecordStorage::open(const rosbag2_storage::StorageOptions& storage_options, rosbag2_storage::storage_interfaces::IOFlag io_flag) {
    storage_options_ = storage_options;
    open_impl(storage_options.uri, io_flag, storage_options.storage_config_uri);
}

void RecordStorage::open(const std::string& uri, rosbag2_storage::storage_interfaces::IOFlag io_flag) {
    open_impl(uri, io_flag, "");
}

void RecordStorage::open_impl(const std::string& uri, rosbag2_storage::storage_interfaces::IOFlag io_flag, const std::string& storage_config_uri) {
    relative_path_ = uri;
    if (relative_path_.length() >= 2) {
        relative_path_.erase(relative_path_.length() - 2);
    }
    switch (io_flag) {
        case rosbag2_storage::storage_interfaces::IOFlag::READ_ONLY: {
            relative_path_ = relative_path_ + RECORD_FILE_EXTENSION;
            record_reader_ = std::make_unique<apollo::cyber::record::RecordReader>(relative_path_);
            break;
        }
        case rosbag2_storage::storage_interfaces::IOFlag::READ_WRITE:
        case rosbag2_storage::storage_interfaces::IOFlag::APPEND: {
            // APPEND does not seem to be used; treat it the same as READ_WRITE
            io_flag = rosbag2_storage::storage_interfaces::IOFlag::READ_WRITE;
            auto opt_header = apollo::cyber::record::HeaderBuilder::GetHeader();
            if (storage_options_.max_bagfile_duration) {
                opt_header.set_segment_interval(storage_options_.max_bagfile_duration * 1000000000ULL);
            }
            if (storage_options_.max_bagfile_size) {
                opt_header.set_segment_raw_size(storage_options_.max_bagfile_size * 1ULL);
            }
            record_writer_ = std::make_unique<apollo::cyber::record::RecordWriter>(opt_header);
            record_writer_->RegisterSplitFileCallback(std::bind(&RecordStorage::on_file_event, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
            relative_path_ = relative_path_ + RECORD_FILE_EXTENSION;
            bool ret = record_writer_->Open(relative_path_, storage_options_.max_files);
            if (!ret) {
                throw std::runtime_error("record writer open failed.");
            }
            break;
        }
    }
    opened_as_ = io_flag;
    metadata_.relative_file_paths = {get_relative_file_path()};
}

/** BaseInfoInterface **/
rosbag2_storage::BagMetadata RecordStorage::get_metadata() {
    std::cerr << "can not call get_metadata " << std::endl;
    return metadata_;
}

std::string RecordStorage::get_relative_file_path() const {
    return relative_path_;
}

uint64_t RecordStorage::get_bagfile_size() const {
    std::cerr << "can not call get_bagfile_size() " << std::endl;
    return 1234;
}

std::string RecordStorage::get_storage_identifier() const {
    return "record";
}

/** BaseReadInterface **/
bool RecordStorage::read_and_enqueue_message() {
    // The recording has not been opened.
    apollo::cyber::record::RecordMessage record_message;
    if (!record_reader_->ReadMessage(&record_message, begin_time_)) {
        return false;
    }

    auto msg = std::make_shared<rosbag2_storage::SerializedBagMessage>();
    msg->time_stamp = record_message.time;
    msg->topic_name = record_message.channel_name;
    msg->serialized_data = rosbag2_storage::make_serialized_message(record_message.content.data(), record_message.content.size());
    // enqueue this message to be used
    next_ = msg;
    return true;
}

bool RecordStorage::has_next() {
    // Have already verified next message and enqueued it for use.
    if (next_) {
        return true;
    }
    return read_and_enqueue_message();
}

std::shared_ptr<rosbag2_storage::SerializedBagMessage> RecordStorage::read_next() {
    if (!has_next()) {
        throw std::runtime_error{"No next message is available."};
    }
    // Importantly, clear next_ via move so that a next message can be read.
    return std::move(next_);
}

std::vector<rosbag2_storage::TopicMetadata> RecordStorage::get_all_topics_and_types() {
    std::set<std::string> topic_list = record_reader_->GetChannelList();
    std::vector<rosbag2_storage::TopicMetadata> out;
    for (const auto& topic : topic_list) {
        rosbag2_storage::TopicMetadata topic_meta_data;
        topic_meta_data.name = topic;
        topic_meta_data.type = record_reader_->GetMessageType(topic);
        topic_meta_data.type_desc = record_reader_->GetProtoDesc(topic);
        out.push_back(topic_meta_data);
    }
    return out;
}

std::shared_ptr<rosbag2_storage::Attachment> RecordStorage::read_Attachment(std::string name) {
    return nullptr;
}

void RecordStorage::get_all_attachments_filepath(std::vector<std::string> &attach_list) {
    /* draft */
    return;
}

/** ReadOnlyInterface **/
void RecordStorage::set_filter(const rosbag2_storage::StorageFilter& storage_filter) {
    storage_filter_ = storage_filter;
}

void RecordStorage::reset_filter() {
    set_filter(rosbag2_storage::StorageFilter());
}

void RecordStorage::seek(const rcutils_time_point_value_t& time_stamp) {
    record_reader_->Reset();
    begin_time_ = time_stamp;
}

/** ReadWriteInterface **/
uint64_t RecordStorage::get_minimum_split_file_size() const {
    return 1024;
}

/** BaseWriteInterface **/
void RecordStorage::write(std::shared_ptr<const rosbag2_storage::SerializedBagMessage> msg) {
    if (topics_.find(msg->topic_name) == topics_.end()) {
        COMMON_LOG_ERROR << msg->topic_name + " has not been created, please create first.";
        return;
    }
    //解析message
    std::vector<char> proto_message;
    std::string proto_name;
    std::string idl_type = topics_[msg->topic_name];
    if ("CmProtoBuf" == idl_type) {
        std::shared_ptr<eprosima::fastrtps::rtps::SerializedPayload_t> payload_ptr = std::make_shared<eprosima::fastrtps::rtps::SerializedPayload_t>();
        payload_ptr->reserve(msg->serialized_data->buffer_length);
        memcpy(payload_ptr->data, msg->serialized_data->buffer, msg->serialized_data->buffer_length);
        payload_ptr->length = msg->serialized_data->buffer_length;

        CmProtoBuf cm_buf;
        CmProtoBufPubSubType cm_pub_sub;
        cm_pub_sub.deserialize(payload_ptr.get(), &cm_buf);
        proto_message = cm_buf.str();
        proto_name = cm_buf.name();
    } else if ("ProtoMethodBase" == idl_type) {
        std::shared_ptr<eprosima::fastrtps::rtps::SerializedPayload_t> payload_ptr = std::make_shared<eprosima::fastrtps::rtps::SerializedPayload_t>();
        payload_ptr->reserve(msg->serialized_data->buffer_length);
        memcpy(payload_ptr->data, msg->serialized_data->buffer, msg->serialized_data->buffer_length);
        payload_ptr->length = msg->serialized_data->buffer_length;

        ProtoMethodBase method_buf;
        ProtoMethodBasePubSubType method_pub_sub;
        method_pub_sub.deserialize(payload_ptr.get(), &method_buf);
        proto_name = method_buf.name();
        proto_message = method_buf.str();
    } else {
        COMMON_LOG_DEBUG << msg->topic_name + " unsupported.";
        return;
    }

    google::protobuf::Message* proto_data = ProtoFactory::getInstance()->GenerateMessageByType(proto_name);
    std::string desc;
    rosbag2_storage_record::internal::RecordProtoFactory::GetDescriptorString(*proto_data, &desc);
    if (regist_proto_type_.find(msg->topic_name) == regist_proto_type_.end()) {
        if (record_writer_->WriteChannel(msg->topic_name, proto_name, desc)) {
            regist_proto_type_[msg->topic_name] = proto_name;
        }
    }
    if (nullptr != proto_data) {
        delete proto_data;
    }
    std::string proto_message_str(proto_message.data(), proto_message.size());
    if (!record_writer_->WriteMessage(msg->topic_name, proto_message_str, msg->time_stamp)) {
        COMMON_LOG_ERROR << "WriteMessage failed.";
    }
}

void RecordStorage::write(const std::vector<std::shared_ptr<const rosbag2_storage::SerializedBagMessage>>& msgs) {
    for (const auto& msg : msgs) {
        write(msg);
    }
}

void RecordStorage::create_topic(const rosbag2_storage::TopicMetadata& topic) {
    if (topics_.find(topic.name) == topics_.end()) {
        topics_[topic.name] = topic.type;
    }
}

void RecordStorage::remove_topic(const rosbag2_storage::TopicMetadata& topic) {
    topics_.erase(topic.name);
}

void RecordStorage::on_file_event(std::string absolute_path, uint64_t na_time, bool is_open) {
    if (is_open) {
        relative_path_ = absolute_path;
    }

    for (auto item : split_file_callback_) {
        item(absolute_path, na_time, is_open);
    }
}
void RecordStorage::add_attach(std::shared_ptr<const rosbag2_storage::Attachment> attachment){

}

void RecordStorage::RegisterSplitFileCallback(std::function<void(std::string, uint64_t, bool)> callback_func) {
    split_file_callback_.push_back(callback_func);
}

}  // namespace rosbag2_storage_plugins

#include "pluginlib/class_list_macros.hpp"  // NOLINT
PLUGINLIB_EXPORT_CLASS(rosbag2_storage_plugins::RecordStorage, rosbag2_storage::storage_interfaces::ReadWriteInterface)
