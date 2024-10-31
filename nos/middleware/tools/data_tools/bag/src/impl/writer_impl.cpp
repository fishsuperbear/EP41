#include <impl/writer_impl.h>
#include "fastdds/rtps/common/SerializedPayload.h"

namespace hozon {
namespace netaos {
namespace bag {

WriterImpl::WriterImpl() {
    _writer = std::make_shared<rosbag2_cpp::Writer>();
    _writer_callbacks.write_split_callback = std::bind(&WriterImpl::onWriterEvent, this, std::placeholders::_1);
    _writer->add_event_callbacks(_writer_callbacks);
};

WriterImpl::~WriterImpl(){_writer=nullptr;};

WriterErrorCode WriterImpl::Open(const WriterOptions& writer_options) {
    //open writer
    _writerOptions = writer_options;
    _writerOptions.max_cache_size = _writerOptions.max_cache_size * 1024 * 1024;
    rosbag2_storage::StorageOptions storageOptions;
    storageOptions.uri = writer_options.output_file_name;
    storageOptions.max_files = writer_options.max_files;
    storageOptions.storage_id = writer_options.storage_id;
    storageOptions.max_bagfile_size = writer_options.max_bagfile_size * 1024 * 1024;
    storageOptions.max_bagfile_duration = writer_options.max_bagfile_duration;
    storageOptions.max_files = writer_options.max_files;
    storageOptions.use_time_suffix = writer_options.use_time_suffix;
    //不使用ros的缓存
    // storageOptions.max_cache_size = recordOptions.max_cache_size;
    rosbag2_cpp::ConverterOptions converter_options{};
    for (const auto& attachment: attachment_list_) {
        _writer->add_attach(attachment);
    }
    _writer->open(storageOptions, converter_options);
    return WriterErrorCode::SUCCESS;
};

void WriterImpl::write(const eprosima::fastrtps::rtps::SerializedPayload_t& payload, const std::string& topic_name , const std::string& idl_tpye, const int64_t& time) {
    _writer->write(payload, topic_name, idl_tpye, time);
}

void WriterImpl::write(const std::string& topic_name, const std::string& proto_name, const std::string& serialized_string, const int64_t& time, const std::string& idl_tpye) {
    CmProtoBuf cm_buf;
    CmProtoBufPubSubType sub_type;
    Proto2cmPrtoBuf(serialized_string, proto_name, cm_buf);
    eprosima::fastrtps::rtps::SerializedPayload_t payload;
    payload.reserve(sub_type.getSerializedSizeProvider(&cm_buf)());
    sub_type.serialize(&cm_buf, &payload);
    _writer->write(payload, topic_name, idl_tpye, time);
}

void WriterImpl::Proto2cmPrtoBuf(const std::string& serialized_data, const std::string& proto_name, CmProtoBuf& proto_idl_data) {
    proto_idl_data.name(proto_name);
    proto_idl_data.str().assign(serialized_data.begin(), serialized_data.end());
}

void WriterImpl::WriterRegisterCallback(const WriterCallback& callback) {
    _register_callbacks.push_back(callback);
};

void WriterImpl::onWriterEvent(rosbag2_cpp::bag_events::BagSplitInfo& info) {
    std::unique_lock<std::mutex> lck(_writer_condition_mtx);
    WriterInfo writer_info;
    if ("" != info.closed_file) {
        writer_info.file_path = info.closed_file;
        writer_info.state = InfoType::FILE_CLOSE;
        writer_info.time = info.na_time;
    }
    if ("" != info.opened_file) {
        writer_info.file_path = info.opened_file;
        writer_info.state = InfoType::FILE_OPEN;
        writer_info.time = info.na_time;
        opened_file_name = info.opened_file;
    }

    for (auto func : _register_callbacks) {
        func(writer_info);
    }
}

void WriterImpl::add_attachment(std::shared_ptr<rosbag2_storage::Attachment> attachment) {
    _writer->add_attach(attachment);
}

void WriterImpl::add_attachment_to_list(std::shared_ptr<rosbag2_storage::Attachment> attachment) {
    attachment_list_.push_back(attachment);
}

std::string WriterImpl::get_opened_file_name() {
    return opened_file_name;
}
}  // namespace bag
}  //namespace netaos
}  //namespace hozon
