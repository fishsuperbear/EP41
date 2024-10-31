#pragma once
#include <dds_recorder_subscriber.h>
#include <pcap.h>
#include <memory>
#include <queue>
#include <rosbag2_cpp/writer.hpp>
#include "packet-someip.h"
#include "recorder.h"

namespace hozon {
namespace netaos {
namespace bag {

class RecorderImpl {

   public:
    RecorderImpl();
    RecordErrorCode Start(const RecordOptions& recordOptions);
    void Stop();
    ~RecorderImpl();
    void SpliteBagNow();
    // void CancleSubscribe(const std::vector<std::string>& topics_list);
    void RecorderRegisterCallback(const RecorderCallback& callback);
    void RegisterPreWriteCallbak(const std::string& topic_name, const PreWriteCallbak& data_handler);

   private:
    RecordOptions _record_options;
    bool _is_list_empty = true;
    bool _is_stop = false;

    rosbag2_cpp::Writer* _writer;
    DDSRecorderSubscriber _subscriber;
    std::queue<BagMessage*> _message_list;
    std::mutex _message_list_mutex;

    //new topic
    std::queue<TopicInfo> _new_valible_topic;
    std::mutex _condition_mtx;
    std::condition_variable _cv;
    uint64_t _total_cache_size = 0;

    std::map<std::string, std::vector<std::string>> topics_map_;
    uint count_ = 0;
    std::thread someip_thread_1;
    std::thread someip_thread_2;
    std::thread someip_thread_3;
    std::thread someip_thread_4;

    //thread
    std::thread write_to_file_thread_;
    std::thread print_thread_;
    std::thread haldle_new_topic_thread_;
    std::thread handle_recorder_event_;

    std::unordered_map<std::string, bool> someip_topic_filter_;
    int someip_topic_filter_mode_;  // 0 - none filter, 1 - select, 2 - exclude
    std::map<uint32_t, std::vector<std::shared_ptr<someip_tp_segment_t>>> someip_tp_map_;

    //data process
    std::map<std::string, PreWriteCallbak> _process_data_func_map;

    //record event
    std::vector<RecorderCallback> _recorder_callbacks;
    std::mutex _recorder_condition_mtx;
    std::condition_variable _recorder_cv;
    std::queue<RecorderInfo> _recorder_info_queue;
    int someip_msg_count_ = 0;
    std::shared_ptr<rosbag2_storage::Attachment> attachment_;
    // writer event
    rosbag2_cpp::bag_events::WriterEventCallbacks _writer_callbacks;
    void OnWriterEvent(rosbag2_cpp::bag_events::BagSplitInfo& info);
    //subscriber event
    void OnSubscriptionMatched(const std::string& topic_name, const TopicState& state);
    void OnNewValidTopic(TopicInfo topicInfo);

    void WriteToFile();
    void HaldleNewTopic();
    void PrintMessageCount();
    void HandleRecorderEventThread();
    void ReadMessage(std::queue<BagMessage*>& temp_message_list);
    void AddMessage(BagMessage* bagMessage);
    // void onNewValidTopic(TopicInfo topicInfo);
    int ReadSomeIpMessage(const RecordOptions& recordOptions);
    void CapPacket(std::string eth_name, std::string filter_exp);
    void ProcessPacket(const u_char* packet, pcap_pkthdr& header);
    void NotifyAdfProcessRecoedStart(std::string topic_name);

    bool MatchTopicName(const someip_hdr_t& header, std::string& topic_name);
    void Write(const std::vector<uint8_t>& message, const std::string& topic_name, const std::string& type_name, const uint64_t& time);
    bool SomeipTpWrite(const char* p_payload, const std::unique_ptr<someip_message_t>& someip_msg, const pcap_pkthdr& header, const std::string& topic_name);
    void SomeipWrite(const char* p_payload, const std::unique_ptr<someip_message_t>& someip_msg, const pcap_pkthdr& header, const std::string& topic_name);
    void SomeipHdrToMessageHdr(const someip_hdr_t& p_someip_hdr, std::unique_ptr<someip_message_t>& someip_msg);

    //
    bool AddAttachments();
    bool AddAttachment(std::string file_path, std::string file_type = "normal", std::string rename = "");
    bool AddAttachmentWithCfgData(std::string rename, std::string *data, std::string file_type = "cfg");
    bool AddCfgAttachment(std::string cfg_key);
    std::shared_ptr<YAML::Node> lidarExtrinsicsParaConvert(std::string& json);
    std::string RePrecision(double num);
};

}  // namespace bag
}  //namespace netaos
}  //namespace hozon
