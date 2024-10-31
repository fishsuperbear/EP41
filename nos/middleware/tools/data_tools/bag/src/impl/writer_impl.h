#pragma once

// #include <map>
// #include <memory>
// #include <string>
// #include <vector>
// #include <fastdds/rtps/common/SerializedPayload.h>
#include <queue>
#include <rosbag2_cpp/writer.hpp>
#include "idl/generated/cm_protobuf.h"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "writer.h"

namespace hozon {
namespace netaos {
namespace bag {

class WriterImpl final {
   public:
    explicit WriterImpl();

    ~WriterImpl();

    WriterErrorCode Open(const WriterOptions& record_options);

    void write(const eprosima::fastrtps::rtps::SerializedPayload_t& payload, const std::string& topic_name , const std::string& idl_tpye, const int64_t& time);
    void write(const std::string& topic_name, const std::string& proto_name, const std::string& serialized_string, const int64_t& time, const std::string& idl_tpye);
    void WriterRegisterCallback(const WriterCallback& callback);
    void add_attachment_to_list(std::shared_ptr<rosbag2_storage::Attachment> attachment);
    std::string get_opened_file_name();
    
   private:
    std::shared_ptr<rosbag2_cpp::Writer>  _writer;
    WriterOptions _writerOptions;
    //writer event
    std::vector<WriterCallback> _register_callbacks;
    std::mutex _writer_condition_mtx;
    // std::condition_variable _writer_cv;
    // std::queue<WriterInfo> _writer_info_queue;
    // writer event
    rosbag2_cpp::bag_events::WriterEventCallbacks _writer_callbacks;
    std::vector<std::shared_ptr<rosbag2_storage::Attachment>> attachment_list_;
    std::string opened_file_name;
    void onWriterEvent(rosbag2_cpp::bag_events::BagSplitInfo& info);
    void Proto2cmPrtoBuf(const std::string& serialized_data, const std::string& proto_name, CmProtoBuf& proto_idl_data);
    void add_attachment(std::shared_ptr<rosbag2_storage::Attachment> attachment);
};

}  // namespace bag
}  //namespace netaos
}  //namespace hozon
