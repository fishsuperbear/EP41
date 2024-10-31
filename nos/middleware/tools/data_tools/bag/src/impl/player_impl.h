#pragma once
#include <dds_player_publisher.h>
#include <memory>
#include <mutex>
#include <queue>
#include <bag_message.hpp>
#include <rosbag2_cpp/clocks/player_clock.hpp>
#include <rosbag2_cpp/reader.hpp>
#include "h265_play_handler.h"
#include "player.h"
#include "someip_player_publisher.h"

namespace hozon {
namespace netaos {
namespace bag {

class PlayerImpl {
   private:
    std::shared_ptr<hozon::netaos::data_tool_common::TopicManager> topic_manager_;

    std::unique_ptr<rosbag2_cpp::Reader> _reader;
    std::mutex _reader_mutex;
    DDSPlayerPublisher _subscriber;
    std::unique_ptr<SomeipPlayerPublisher> _uptr_someip_publisher;
    std::queue<BagMessage*> _messageList;
    std::mutex _messageListMutex;
    std::map<std::string, std::string> _topicTypeMap;
    PlayerOptions _playerOptions;

    //h265 data list
    std::queue<BagMessage*> _h265MessageList;
    std::mutex _h265Mutex;
    std::unique_ptr<H265PlayHandler> _h265_play;

    rcutils_time_point_value_t _first_message_time;
    rcutils_time_point_value_t _starting_time;
    rcutils_time_point_value_t _ending_time;
    rcutils_time_point_value_t file_ending_time;
    std::unique_ptr<rosbag2_cpp::PlayerClock> _clock;
    std::mutex _clock_mutex;
    bool _isSrorageEmpty = false;
    bool _isStop = false;
    bool _isPopPause = false;
    bool _isPushPause = false;
    int64_t _pause_time;
    void prepare_publishers();
    void publicMessage();
    void newThread();
    BagMessage* PopMessage();
    void PushMessage(BagMessage* message);
    void PushH265Message(BagMessage* message);
    bool getTopicType(std::vector<std::string> topics);
    std::map<std::string, std::vector<std::string>> topics_map_;
    bool SetRate(double rate);
    bool InitClockAndReaderTime();
    void PublishH265Data();
    bool PrepareSomeipPublishers(const PlayerOptions& playerOptions);
    void NotifyAdfProcessPlayerStart(const std::map<std::string, std::string>& topic_map);
    void ReadMessage();
    bool ifPlayPointCloud(PlayerOptions& playerOptions);
    bool _is_someip = false;
    bool read_Attachment(const std::string file_name, std::string& data);
    bool if_dds_prepare_ = false;
    bool if_someip_prepare_ = false;
    bool auto_completed_ = false;
    bool inited_ = false;
    std::thread read_bag_th_;
    std::thread h265_pub_th_;
    std::thread normal_pub_th_;
    std::string lidar_conf_path;

   public:
    PlayerImpl();
    void Start(const PlayerOptions& playerOptions);
    void Stop();
    void Pause();
    void Resume();
    void FastForward(int interval_tiem);
    void Rewind(int interval_tiem);
    ~PlayerImpl();
};

}  // namespace bag
}  //namespace netaos
}  //namespace hozon
