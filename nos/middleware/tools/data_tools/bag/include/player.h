#pragma once
#include <iostream>
#include <memory>
#include <vector>

namespace hozon {
namespace netaos {
namespace bag {

class PlayerImpl;

struct PlayerOptions {
   public:
    // bag file path
    std::string uri = "";
    // bag file format
    std::string storage_id = "mcap";
    // only play topics related to "topics" variables. if "topics" is empty, will play all topics in bag
    std::vector<std::string> topics;
    // not play the specified topic
    std::vector<std::string> exclude_topics;
    // when "force=true", whether the topic exists or not, it will be played
    bool force = false;
    // Loop Playback
    bool loop = false;
    // The queue size indiciates how many messages can maximally be hold in cache
    // before these being play. value rang from 1 to 1000000.
    uint64_t queue_size = 1000;
    // Starting from the specified time point. in seconds.
    int64_t start_offset = 0;
    //speify the playback rate. Default is 1(original rate)
    float rate = 1.0;
    // Sleep before play. Negative durations invalid. Will delay at the beginning of each loop. In seconds. Default 0.
    int delay = 0;
    std::string begin = "";
    std::string end = "";
    // convert h265 to yuv and play. Defaul 0. 0:don't play yuv; 4:4-way surround view; 7:2 forward views, 4 peripheral views, and 1 tail view; 11:all view.
    std::string h265 = "";
    // whether play topic in protomethod type. Default false.
    bool protomethod = false;
    // someip 回放配置路径，包括网络配置等。
    std::string lidar_conf_path;

    //The following parameters are not yet supported
    bool play_all_topic = false;
    bool start_paused = false;
    bool update_pubtime = false;
    bool force_play_pointcloud = false;
    //The time interval between a fast forward or rewind. in seconds. Default 3s;
    // int64_t seek_interval = 3;
    // std::string duration = "";
    // bool adjust_clock = false;
    // std::string exclude_reg = "";
    // bool quiet = false;
    // uint64_t loop_times = 0;
    // bool skip_empty = false;
    // std::vector<std::string> pause_topics;
    // std::vector<std::string> topic_remapping_options = {};
    // double clock_publish_frequency = 0.0;
    // bool disable_keyboard_controls = false;
    // bool show_help_info = false;
};

class Player {
   public:
    Player();
    void Start(const PlayerOptions& playerOptions);
    void Stop();
    void Pause();
    void Resume();
    void FastForward(int interval_tiem);
    void Rewind(int interval_tiem);
    ~Player();

   private:
    std::unique_ptr<PlayerImpl> player_impl_;
};

}  // namespace bag
}  //namespace netaos
}  //namespace hozon
