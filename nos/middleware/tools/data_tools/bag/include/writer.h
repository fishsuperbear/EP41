#pragma once
// #include <map>
#include <functional>
#include <memory>

// #include <string>
// #include <vector>
// #include <fastdds/rtps/common/SerializedPayload.h>

namespace hozon {
namespace netaos {
namespace bag {

struct WriterOptions {
   public:
    // 目标文件格式。支持:"mcap","record"
    std::string storage_id = "mcap";
    // output file name
    // 输出文件名，例如:/home/CameraBag. 若不指定，默认以当前时间为包名。
    std::string output_file_name = "";
    // The maximum size a bagfile can be, in Mb, before it is split. Default is 2GB.
    // 录制文件的最大size，超过这个大小会进行分包。单位：Mb，默认大小为2G
    uint64_t max_bagfile_size = 2048;
    // The maximum duration a bagfile can be, in seconds, before it is split. Default is 1h.
    // 录制文件的最大时长，超过这个时长会进行分包。单位: s(秒)，默认时长为1h
    uint64_t max_bagfile_duration = 60 * 60ULL;
    // Maximum File Retained. old file will be remove. value rang [1, 864000]. Default 864000.
    // 指定保留录制包的个数，若超出此限制，旧的包会被删除。取值范围[1, 864000]，默认为864000
    uint64_t max_files = 86400;
    //缓存的大小，单位M，默认200M
    uint64_t max_cache_size = 200;
    //指定包名时是否带时间后缀，默认带
    bool use_time_suffix = true;
};

enum class WriterErrorCode {
    SUCCESS = 0,
    OPEN_FILE_FAILED = 1  //打开文件失败
};

class WriterImpl;

enum class InfoType : uint8_t {
    FILE_OPEN = 1,           // 打开bag file
    FILE_CLOSE = 2,          // 关闭当前bag file
    TOPIC_SUBSCRIBE = 3,     // 订阅toic
    TOPIC_DESCONNECTED = 4,  // 已订阅的topic退出
};

struct WriterInfo {
    InfoType state;          //状态
    std::string topic_name;  //topic名
    uint64_t time = 0;       //纳秒时间戳
    std::string file_path;   //文件绝对路径
};

using WriterCallback = std::function<void(WriterInfo&)>;

class Writer final {
   public:
    explicit Writer();

    ~Writer();

    /**
   * Throws if file could not be opened.
   * This must be called before any other function is used.
   * The rosbag is automatically closed on destruction.
   *
   * \param uri bag path
   */
    WriterErrorCode Open(const WriterOptions& record_options);

    /**
   * Closing the reader instance.
   */
    template <typename ProtoType>
    void WriteEventProtoMessage(const std::string& topic_name, const ProtoType& proto_data, const int64_t& time) {
        std::string proto_name = proto_data.GetTypeName();
        std::string serialized_string;
        proto_data.SerializeToString(&serialized_string);
        write(topic_name, proto_name, serialized_string, time, "CmProtoBuf");
    };

    void WriterRegisterCallback(const WriterCallback& callback);

   private:
    void write(const std::string& topic_name, const std::string& proto_name, const std::string& serialized_string, const int64_t& time, const std::string& idl_tpye);
    std::unique_ptr<WriterImpl> writer_impl_;
};

}  // namespace bag
}  //namespace netaos
}  //namespace hozon
