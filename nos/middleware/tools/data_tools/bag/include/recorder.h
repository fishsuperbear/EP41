#pragma once
#include <functional>
#include <iostream>
#include <memory>
#include <vector>
#include "bag_message.hpp"

namespace hozon {
namespace netaos {
namespace bag {

class RecorderImpl;

struct RecordOptions {
   public:
    // 目标文件格式。支持:"mcap","record"
    std::string storage_id = "mcap";
    // The maximum size a bagfile can be, in Mb, before it is split. Default is 2GB.
    // 录制文件的最大size，超过这个大小会进行分包。单位：Mb，默认大小为2G
    uint64_t max_bagfile_size = 2048;
    // The maximum duration a bagfile can be, in seconds, before it is split. Default is 1h.
    // 录制文件的最大时长，超过这个时长会进行分包。单位: s(秒)，默认时长为1h
    uint64_t max_bagfile_duration = 60 * 60ULL;
    // The cache size indiciates how many messages can maximally be hold in cache
    // before these being written to disk. values range from 1 to 1000000.s. Default is 1000.
    // 缓存队列的大小，默认值1000.
    uint64_t queue_size = 1000;
    // Record only topics related to "topics" variables. if "topics" is empty, will record all topic
    // 需要录制的topic，若不指定默认录制全部的topic
    std::vector<std::string> topics;
    //not record the specified topic
    // 录制指定topic以外的全部topic
    std::vector<std::string> exclude_topics;
    //是否录制method topic
    bool method = false;
    // output file name
    // 输出文件名，例如:/home/CameraBag. 若不指定，默认以当前时间为包名。
    std::string output_file_name = "";
    // Maximum File Retained. old file will be remove. value rang [1, 864000]. Default 864000.
    // 指定保留录制包的个数，若超出此限制，旧的包会被删除。取值范围[1, 864000]，默认为864000
    uint64_t max_files = 86400;
    //缓存的大小，单位M，默认200M
    uint64_t max_cache_size = 200;
    //指定包名时是否带时间后缀，默认带
    bool use_time_suffix = true;
    // 需要录制的附件
    std::vector<std::string> attachments;

    std::string someip_network = "test";

    //The following parameters are not yet supported
    bool record_all = false;

    // // 是否录制雷达配置文件 ATD128P.dat
    // bool record_ATD128P = false;

    // // 是否录制雷达外参文件 lidar_extrinsics.yaml
    // bool record_lidar_extrinsics = false;
    
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

enum RecordErrorCode {
    SUCCESS = 0,
    INIT_SUBSCRIBER_FAILED = 1,   //通信初始化失败
    MESSAGE_DISCARDED_BY_APP = 2  // message数据被app丢弃

};

enum class RecorderState : uint8_t {
    FILE_OPEN = 1,           // 打开bag file
    FILE_CLOSE = 2,          // 关闭当前bag file
    TOPIC_SUBSCRIBE = 3,     // 订阅toic
    TOPIC_DESCONNECTED = 4,  // 已订阅的topic退出
};

struct RecorderInfo {
    RecorderState state;     //状态
    std::string topic_name;  //topic名
    uint64_t time = 0;       //纳秒时间戳
    std::string file_path;   //文件绝对路径
};

using RecorderCallback = std::function<void(RecorderInfo&)>;
using PreWriteCallbak = std::function<RecordErrorCode(BagMessage& cur_msg, std::vector<BagMessage>& pre_msg, std::vector<BagMessage>& post_msg)>;

class Recorder {
   public:
    Recorder();
    RecordErrorCode Start(const RecordOptions& record_options);
    RecordErrorCode Stop();
    //调用后会立即分包，不管max_bagfile_duration、max_bagfile_size是否满足
    void SpliteBagNow();
    ~Recorder();
    void RecorderRegisterCallback(const RecorderCallback& callback);
    void RegisterPreWriteCallbak(const std::string& topic_name, const PreWriteCallbak& data_handler);

   private:
    std::unique_ptr<RecorderImpl> recorder_impl_;
};

}  // namespace bag
}  //namespace netaos
}  //namespace hozon
