#pragma once
#include <iostream>
#include <memory>
#include <vector>

namespace hozon {
namespace netaos {
namespace bag_py {

class RecorderImpl;

struct RecordOptions {
   public:
    // storage format
    std::string storage_id = "mcap";
    // The maximum size a bagfile can be, in Mb, before it is split.
    // Default is 2GB.
    uint64_t max_bagfile_size = 2048;
    // The maximum duration a bagfile can be, in seconds, before it is split.
    // Default is 1h.
    uint64_t max_bagfile_duration = 60 * 60ULL;
    // The cache size indiciates how many messages can maximally be hold in cache
    // before these being written to disk. values range from 1 to 1000000.s
    uint64_t queue_size = 1000;
    //Record only topics related to "topics" variables. if "topics" is empty, will record all topic
    std::vector<std::string> topics;
    //not record the specified topic
    std::vector<std::string> exclude_topics;
    //output file name
    std::string output_file_name = "";
    //Maximum File Retained. old file will be remove. value rang [1, 864000]. Default 864000.
    uint64_t max_files = 86400;

    //The following parameters are not yet supported
    bool record_all = false;
    // bool show_help_info = false;
    // uint64_t size = 0;
    // std::string duration = "";
    // std::string exclude_reg = "";
    // uint64_t max_file_buffer = 2;
    // std::string compress_type = "";
    // uint64_t skip_frame_num = 0;
    // std::string encrypt_type = "";
    // std::string key_file = "";
};

enum RecordErrorCode { INIT_SUBSCRIBER_FAILED = -1, SUCCESS = 0 };

class Recorder {
   public:
    Recorder();
    RecordErrorCode Start(const RecordOptions& record_options);
    void Stop();
    ~Recorder();

   private:
    std::unique_ptr<RecorderImpl> recorder_impl_;
};

}  // namespace bag_py
}  //namespace netaos
}  //namespace hozon
