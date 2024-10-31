#include "impl/player_impl.h"
#include <sys/stat.h>
#include <chrono>
#include "adf_lite_utile.h"
#include "cost_calc.h"
#include "message_process.h"
#include "player.h"
#include "process_utility.h"
#include "rcpputils/filesystem_helper.hpp"
#include "rosbag2_cpp/clocks/time_controller_clock.hpp"
#include "topic_manager.hpp"

namespace hozon {
namespace netaos {
namespace bag {

using namespace hozon::netaos::data_tool_common;

// bool isSomeipTopic(std::string topic_name) {
//     for (size_t i = 0; i < SOMEIP_TOPIC_NUM; i++)
//     {
//         if (std::string::npos != topic_name.find(someip_topic_name[i])) {
//             return true;
//         }
//     }
//     return false;
// }

bool isSomeipMessage(std::string type) {
    return (type == "someip_message_t");
}

PlayerImpl::PlayerImpl()
    : topic_manager_(std::make_shared<hozon::netaos::data_tool_common::TopicManager>()), _subscriber(topic_manager_) {
    _reader = std::make_unique<rosbag2_cpp::Reader>();
    if (!topic_manager_->Init(true)) {
        BAG_LOG_ERROR << "TopicManager init failed!";
    }
};

PlayerImpl::~PlayerImpl() {

    {
        std::lock_guard<std::mutex> lk(_messageListMutex);
        while (_messageList.size() > 0) {
            BagMessage* bagMessage = _messageList.front();
            _messageList.pop();
            delete bagMessage;
        }
    }

    {
        std::lock_guard<std::mutex> lk(_h265Mutex);
        while (_h265MessageList.size() > 0) {
            BagMessage* bagMessage = _h265MessageList.front();
            _h265MessageList.pop();
            delete bagMessage;
        }
    }

    topic_manager_->DeInit();
};

void PlayerImpl::Start(const PlayerOptions& playerOptions) {
    // ProcessUtility::SetThreadName("play_main");

    _is_someip = false;
    if_dds_prepare_ = false;
    if_someip_prepare_ = false;
    auto_completed_ = false;
    inited_ = false;
    lidar_conf_path = playerOptions.lidar_conf_path;

    struct InitFlagWrapper {
        InitFlagWrapper(bool& flag) : flag_(flag) {}

        ~InitFlagWrapper() { flag_ = true; }

        bool& flag_;
    };

    InitFlagWrapper init_flag_wrapper(inited_);

    if (!rcpputils::fs::exists(playerOptions.uri)) {
        CONCLE_BAG_LOG_ERROR << playerOptions.uri << " does not exist!";
        BAG_LOG_ERROR << playerOptions.uri << " does not exist!";
        inited_ = true;
        return;
    }
    _playerOptions = playerOptions;

    if (_playerOptions.delay > 0) {
        std::cout << "delay " << _playerOptions.delay << "s before play." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(_playerOptions.delay));
    }

    // get topic type and set reader filter
    if (!getTopicType(playerOptions.topics)) {
        inited_ = true;
        return;
    }

    // public h265 messages thread
    // h265 init will cost long time, so it need before InitClockAndReaderTime().
    if (_playerOptions.h265.size() > 0) {
        // init will block thread
        _h265_play.reset(new H265PlayHandler(_playerOptions.h265, topic_manager_));
        h265_pub_th_ = std::thread(&PlayerImpl::PublishH265Data, this);
    }

    //init clock and set the reader start time
    if (!InitClockAndReaderTime()) {
        inited_ = true;
        return;
    }

    //read messages thread
    read_bag_th_ = std::thread(&PlayerImpl::newThread, this);

    // std::cout << "\nHit 'space' to toggle paused, '<-' to fast forward, '->' to rewind.\n" << std::endl;
    //prepare publishers and public messge
    _subscriber.SetUpdatePubTimeFlag(_playerOptions.update_pubtime);

    if_dds_prepare_ =
        _subscriber.InitWriters(_topicTypeMap, playerOptions.protomethod, (_playerOptions.h265.size() == 0));

    // prepare someip subscriber
    if_someip_prepare_ = PrepareSomeipPublishers(playerOptions);

    // if (_subscriber.prepareWriters(_topicTypeMap)) {
    if (if_dds_prepare_ || if_someip_prepare_) {
        normal_pub_th_ = std::thread(&PlayerImpl::ReadMessage, this);
    } else {
        auto_completed_ = true;
    }

    inited_ = true;

    while (!_isStop && !auto_completed_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (!_isStop && auto_completed_) {
        Stop();
    }
};

void PlayerImpl::ReadMessage() {
    publicMessage();
    while (_playerOptions.loop && !_isStop) {

        {
            std::lock_guard<std::mutex> lk(_reader_mutex);
            _reader->seek(_starting_time);
            _clock->jump(_first_message_time);
            _isSrorageEmpty = false;
        }
        if (_playerOptions.delay > 0) {
            std::cout << "delay " << _playerOptions.delay << "s before play." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(_playerOptions.delay));
        }
        publicMessage();
        // wait_for_filled_queue
        // std::this_thread::sleep_for(std::chrono::milliseconds(2));
        if (_isSrorageEmpty) {
            if (!_playerOptions.loop) {
                break;
            }
        }
    }

    if (!_isStop) {
        auto_completed_ = true;
    }
}

bool PlayerImpl::InitClockAndReaderTime() {
    //init clock and set the reader start time
    {
        std::lock_guard<std::mutex> lk(_reader_mutex);
        // keep reader open until player is destroyed
        auto metadata = _reader->get_metadata();
        _starting_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(metadata.starting_time.time_since_epoch()).count();
        _ending_time = file_ending_time =
            _starting_time + std::chrono::duration_cast<std::chrono::nanoseconds>(metadata.duration).count();
        //set start time
        if ("" != _playerOptions.begin) {
            std::tm tmTime = {};
            std::istringstream iss(_playerOptions.begin);
            iss >> std::get_time(&tmTime, "%Y-%m-%d-%H:%M:%S");
            if (iss.fail()) {
                throw std::runtime_error("Failed to parse time string:" + _playerOptions.begin);
            }
            std::chrono::system_clock::time_point timePoint =
                std::chrono::system_clock::from_time_t(std::mktime(&tmTime));
            // 计算时间点相对于基准时间点的纳秒数
            std::chrono::nanoseconds duration =
                std::chrono::duration_cast<std::chrono::nanoseconds>(timePoint.time_since_epoch());
            if (duration.count() > _starting_time) {
                _starting_time = duration.count();
            }
        } else if (_playerOptions.start_offset < 0) {
            BAG_LOG_WARN << "Invalid start offset value: " << static_cast<double>(_playerOptions.start_offset)
                         << ". Negative start offset ignored.";
        } else if (_playerOptions.start_offset > 0) {
            // If a non-default (positive) starting time offset is provided in PlayOptions,
            // then add the offset to the starting time obtained from reader metadata
            auto nanoseconds =
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(_playerOptions.start_offset));
            _starting_time += nanoseconds.count();
        }

        //set end time
        if ("" != _playerOptions.end) {
            std::tm tmTime = {};
            std::istringstream iss(_playerOptions.end);
            iss >> std::get_time(&tmTime, "%Y-%m-%d-%H:%M:%S");
            if (iss.fail()) {
                throw std::runtime_error("Failed to parse time string:" + _playerOptions.end);
            }
            std::chrono::system_clock::time_point timePoint =
                std::chrono::system_clock::from_time_t(std::mktime(&tmTime));
            // 计算时间点相对于基准时间点的纳秒数
            std::chrono::nanoseconds duration =
                std::chrono::duration_cast<std::chrono::nanoseconds>(timePoint.time_since_epoch());
            if (duration.count() < _ending_time) {
                _ending_time = duration.count();
            }
        }

        if (_starting_time >= _ending_time) {
            CONCLE_BAG_LOG_ERROR << "End time earlier than start time. Please check.";
            BAG_LOG_ERROR << "End time earlier than start time. Please check.";
            return false;
        }

        _clock = std::make_unique<rosbag2_cpp::TimeControllerClock>(_starting_time, std::chrono::steady_clock::now,
                                                                    std::chrono::milliseconds{100},
                                                                    _playerOptions.start_paused);
        SetRate(_playerOptions.rate);
        _reader->seek(_starting_time);
        // _clock->jump(_starting_time);
    }

    //set the clock to the first message time
    {
        std::lock_guard<std::mutex> lk(_reader_mutex);
        if (_reader->has_next()) {
            std::shared_ptr<rosbag2_storage::SerializedBagMessage> message = _reader->read_next();
            _first_message_time = message->time_stamp;
            _clock->jump(_first_message_time);
        }
        //reset the reader
        _reader->seek(_starting_time);
    }
    return true;
}

void PlayerImpl::Stop() {
    _isStop = true;

    // Wait until init completed. It will core if stop when it is initiaizeing.
    if (!inited_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    }

    //stop adf-lite
    if (topics_map_.size() > 0) {
        //请求停止adf-lite
        RequestCommand(topics_map_, "play", false);
    }

    if (normal_pub_th_.joinable()) {
        normal_pub_th_.join();
    }

    if (h265_pub_th_.joinable()) {
        h265_pub_th_.join();
    }

    if (read_bag_th_.joinable()) {
        read_bag_th_.join();
    }

    {
        std::lock_guard<std::mutex> lk(_messageListMutex);
        while (_messageList.size() > 0) {
            BagMessage* bagMessage = _messageList.front();
            _messageList.pop();
            delete bagMessage;
        }
    }

    {
        std::lock_guard<std::mutex> lk(_h265Mutex);
        while (_h265MessageList.size() > 0) {
            BagMessage* bagMessage = _h265MessageList.front();
            _h265MessageList.pop();
            delete bagMessage;
        }
    }

    if (_h265_play) {
        _h265_play->Stop();
    }
    _h265_play = nullptr;
    _subscriber.DeinitWriters();
    topic_manager_->DeInit();

    return;
};

void PlayerImpl::PushMessage(BagMessage* message) {

    while (!_isStop) {
        {
            std::lock_guard<std::mutex> lk(_messageListMutex);
            if (_messageList.size() < _playerOptions.queue_size || _isPushPause) {
                _messageList.push(message);
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
}

void PlayerImpl::PushH265Message(BagMessage* message) {
    while (!_isStop) {
        {
            std::lock_guard<std::mutex> lk(_h265Mutex);
            if (_h265MessageList.size() < _playerOptions.queue_size || _isPushPause) {
                _h265MessageList.push(message);
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
}

bool PlayerImpl::ifPlayPointCloud(PlayerOptions& playerOptions) {

    bool if_force_play = playerOptions.force_play_pointcloud;
    std::string buffer;
    std::string extrinsics_file_name = "conf_calib_lidar/roof_lidar_params";
    bool if_has_extrubsucs = read_Attachment(extrinsics_file_name, buffer);
    return (if_force_play || if_has_extrubsucs);
}

void PlayerImpl::newThread() {

    ProcessUtility::SetThreadName("player_pub_main");
    while (!_isStop) {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        if (_isSrorageEmpty || _isPushPause) {
            continue;
        }
        //read message to queue
        {
            std::lock_guard<std::mutex> lk(_reader_mutex);
            if (_reader->has_next()) {
                std::shared_ptr<rosbag2_storage::SerializedBagMessage> message = _reader->read_next();
                if (message->time_stamp > _ending_time) {
                    _reader->seek(file_ending_time + 2);
                    continue;
                }
                BagMessage* tempMessage = new BagMessage();
                std::string topic_name = message->topic_name;

                if (topic_name == "/soc/rawpointcloud") {
                    // 指定了激光雷达配置文件
                    if (!lidar_conf_path.empty()) {
                        if (MessageProcess::Instance(lidar_conf_path).Process(message, *tempMessage, topic_name)) {
                            BAG_LOG_WARN << "Process pointcloud fail!";
                            delete tempMessage;
                            continue;
                        }
                    }
                    // 未指定激光雷达配置文件，判断是否要继续回放
                    else if (!ifPlayPointCloud(_playerOptions)) {
                        std::cout << "Nether assigned lidar extrinsics file, nor contained in mcap. Stop play pointcloud. Please use --no-motion-comp to force play." << std::endl;
                        delete tempMessage;
                        _isSrorageEmpty = true;
                        continue;
                    }
                    // 未指定激光雷达配置文件，则从mcap包里读取
                    else if (0 != MessageProcess::Instance(_playerOptions.uri, _playerOptions.storage_id)
                                      .Process(message, *tempMessage, topic_name)) {
                        BAG_LOG_WARN << "Process pointcloud fail!";
                        delete tempMessage;
                        continue;
                    }
                } else {
                    tempMessage->topic = message->topic_name;
                    tempMessage->type = _topicTypeMap[message->topic_name];
                    tempMessage->time = message->time_stamp;
                    tempMessage->data.m_payload->reserve(message->serialized_data->buffer_length);
                    memcpy(tempMessage->data.m_payload->data, message->serialized_data->buffer,
                           message->serialized_data->buffer_length);
                    tempMessage->data.m_payload->length = message->serialized_data->buffer_length;
                }

                if (_playerOptions.h265.size() > 0 && h265_set.find(message->topic_name) != h265_set.end()) {
                    PushH265Message(tempMessage);
                } else {
                    PushMessage(tempMessage);
                }
            } else {
                _isSrorageEmpty = true;
            }
        }
    }
};

void PlayerImpl::publicMessage() {
    BagMessage* bagMessage = nullptr;
    while (!_isStop) {
        if (_isSrorageEmpty && (_messageList.size() == 0) && (_h265MessageList.size() == 0)) {
            BAG_LOG_INFO << "play over!";
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(500));
        bagMessage = PopMessage();
        while (bagMessage) {
            {
                // Do not move on until sleep_until returns true
                // It will always sleep, so this is not a tight busy loop on pause
                std::lock_guard<std::mutex> lk(_clock_mutex);
                while (!_clock->sleep_until(bagMessage->time)) {}
            }

            if (_isStop) {
                break;
            }

            if (isSomeipMessage(bagMessage->type)) {
                // 回放someip
                if (_is_someip) {
                    _uptr_someip_publisher->Publish(bagMessage);
                } else {
                    BAG_LOG_INFO << "someip_publisher not init, do not publish someip message!";
                }
            } else {
                _subscriber.Publish(bagMessage);
            }

            // _clock->jump(bagMessage->time);
            //判断是否暂停
            while (_isPopPause) {
                if (_isStop) {
                    break;
                }
                _pause_time = bagMessage->time;
                std::this_thread::sleep_for(std::chrono::microseconds(500));
            }
            std::cout << std::fixed << std::setprecision(5)
                      << std::chrono::duration_cast<std::chrono::duration<double>>(
                             std::chrono::nanoseconds(bagMessage->time))
                             .count()
                      << "/";
            std::cout << std::fixed
                      << std::chrono::duration_cast<std::chrono::duration<double>>(
                             std::chrono::nanoseconds(_ending_time))
                             .count()
                      << "\r";
            std::cout.flush();
            //判断是否停止
            if (_isStop) {
                delete bagMessage;
                break;
            }
            delete bagMessage;
            bagMessage = PopMessage();
        }
    }
}

BagMessage* PlayerImpl::PopMessage() {
    std::lock_guard<std::mutex> lk(_messageListMutex);
    BagMessage* bagMessage = nullptr;
    if (_messageList.size() > 0) {
        bagMessage = _messageList.front();
        _messageList.pop();
    }
    return bagMessage;
};

bool PlayerImpl::getTopicType(std::vector<std::string> topics) {
    //get storage file topics
    rosbag2_storage::StorageOptions storage_option;
    storage_option.uri = _playerOptions.uri;
    storage_option.storage_id = _playerOptions.storage_id;
    rosbag2_cpp::ConverterOptions converter_options{};
    if (!_reader->open(storage_option, converter_options)) {
        CONCLE_BAG_LOG_ERROR << "open " << storage_option.uri << " failed.";
        BAG_LOG_ERROR << "open " << storage_option.uri << " failed.";
        return false;
    }

    //获取mcap包中的所有topic
    std::vector<rosbag2_storage::TopicMetadata> storageTopics = _reader->get_all_topics_and_types();

    //storageTopics去掉exclude_topics中的topic
    for (auto exclude_topic : _playerOptions.exclude_topics) {
        for (auto it = storageTopics.begin(); it != storageTopics.end();) {
            if ((*it).name == exclude_topic) {
                it = storageTopics.erase(it);  // 删除元素并获取下一个有效迭代器
            } else {
                ++it;
            }
        }
    }

    if (topics.size() > 0) {
        //播放指定的topics
        //不能同时播放/soc/rawpointcloud、/soc/pointcloud
        if ((topics.end() != std::find(topics.begin(), topics.end(), "/soc/rawpointcloud")) &&
            (topics.end() != std::find(topics.begin(), topics.end(), "/soc/pointcloud"))) {
            CONCLE_BAG_LOG_ERROR << "'/soc/rawpointcloud' and '/soc/pointcloud' can't be playing at the same time.";
            BAG_LOG_ERROR << "'/soc/rawpointcloud' and '/soc/pointcloud' can't be playing at the same time.";
            return false;
        }
        for (size_t i = 0; i < topics.size(); i++) {
            //check if the topic is in the bag
            bool isTopicInstorage = false;
            for (size_t j = 0; j < storageTopics.size(); j++) {
                if (storageTopics[j].name == topics[i]) {
                    isTopicInstorage = true;
                    _topicTypeMap[topics[i]] = storageTopics[j].type;
                }
            }
            if (!isTopicInstorage) {
                CONCLE_BAG_LOG_WARN << "no topic: " << topics[i] << " in storage!";
                BAG_LOG_WARN << "no topic: " << topics[i] << " in storage!";
            }
        }
        NotifyAdfProcessPlayerStart(_topicTypeMap);
    } else {
        //Play all topics in bag
        //不能同时播放/soc/rawpointcloud、/soc/pointcloud
        bool raw_point_exist = false;
        bool point_exist = false;
        for (auto item : storageTopics) {
            if ("/soc/rawpointcloud" == item.name) {
                raw_point_exist = true;
            } else if ("/soc/pointcloud" == item.name) {
                point_exist = true;
            }
        }
        if (raw_point_exist && point_exist) {
            CONCLE_BAG_LOG_ERROR << "'/soc/rawpointcloud' and '/soc/pointcloud' can't be playing at the same time.";
            BAG_LOG_ERROR << "'/soc/rawpointcloud' and '/soc/pointcloud' can't be playing at the same time.";
            return false;
        }

        for (auto topicInfo : storageTopics) {
            _topicTypeMap[topicInfo.name] = topicInfo.type;
        }
        NotifyAdfProcessPlayerStart(_topicTypeMap);
    }

    if (_topicTypeMap.size() == 0) {
        CONCLE_BAG_LOG_WARN << "no topic to play";
        BAG_LOG_WARN << "no topic to play";
        return false;
    }

    //设置读包filter
    std::vector<std::string> public_topic;
    for (auto item : _topicTypeMap) {
        public_topic.push_back(item.first);
    }
    rosbag2_storage::StorageFilter storage_filter;
    storage_filter.topics = public_topic;
    _reader->set_filter(storage_filter);

    // /soc/rawpointcloud数据，以/soc/pointcloud发布
    auto it = _topicTypeMap.find("/soc/rawpointcloud");
    if (it != _topicTypeMap.end()) {
        auto value = it->second;
        _topicTypeMap.erase(it);
        _topicTypeMap.insert(std::make_pair("/soc/pointcloud", value));
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));
    //检查是否已有topic在播放
    auto validTopics = topic_manager_->GetTopicInfo();  //获取正在播放的topics
    std::vector<std::string> exist_topics_list;
    for (auto topicInfo : _topicTypeMap) {
        if (validTopics.find(topicInfo.first) != validTopics.end()) {
            exist_topics_list.push_back(topicInfo.first);
        }
    }
    if (exist_topics_list.size() > 0 && !_playerOptions.force) {
        for (auto topic_name : exist_topics_list) {
            std::cerr << "\033[31m"
                      << "Error: topic:" << topic_name
                      << " already exists! please Kill the other thread or use -f option."
                      << "\033[0m" << std::endl;
        }
        Stop();
        return false;
    } else {
        for (auto topic_name : exist_topics_list) {
            std::cout << "\033[33m"
                      << "Warning: topic:" << topic_name << " already exists! will be force play"
                      << "\033[0m" << std::endl;
        }
    }
    return true;
};

void PlayerImpl::NotifyAdfProcessPlayerStart(const std::map<std::string, std::string>& topic_map) {
    // 将topics按process进行分类
    for (auto item : topic_map) {
        std::string process_name;
        int32_t res = hozon::netaos::data_tool_common::GetPartFromCmTopic(item.first, 1, process_name);
        if (res == -1) {
            continue;
        }
        BAG_LOG_DEBUG << "topic :" << item.first << " process_name:" << process_name;
        if (process_name.size() > 0) {
            topics_map_[process_name].push_back(item.first);
        }
    }

    if (topics_map_.size() > 0) {
        //发送adf-lite request
        RequestCommand(topics_map_, "play", true);
    }
    return;
}

void PlayerImpl::Pause() {
    if (!_isPopPause) {
        _isPopPause = true;
        if (_h265_play) {
            _h265_play->SetPause(true);
        }
        std::lock_guard<std::mutex> lk(_clock_mutex);  //等目前正在发送的这帧消息发送完，即sleep_util()被唤醒
    }
}

void PlayerImpl::Resume() {
    if (_isPopPause) {
        {
            std::lock_guard<std::mutex> lk(_clock_mutex);  //等目前正在发送的这帧消息发送完，即sleep_util()被唤醒
            _clock->jump(_pause_time);
            _isPopPause = false;
            if (_h265_play) {
                _h265_play->SetPause(false);
            }
        }
    }
}

bool PlayerImpl::SetRate(double rate) {
    bool ok = _clock->set_rate(rate);
    if (ok) {
        BAG_LOG_DEBUG << "Set rate to " << rate;
    } else {
        BAG_LOG_ERROR << "Failed to set rate to invalid value " << rate;
    }
    return ok;
}

void PlayerImpl::FastForward(int interval_tiem) {
    Pause();              //暂停发布
    _isPushPause = true;  //停止往list写数据
    rcutils_time_point_value_t now_tem =
        _pause_time + std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(interval_tiem)).count();
    {
        std::lock_guard<std::mutex> lk(_reader_mutex);
        {
            //清空缓存队列
            std::lock_guard<std::mutex> lk(_messageListMutex);
            std::queue<BagMessage*> emptyQueue;
            std::swap(_messageList, emptyQueue);
            _isSrorageEmpty = false;
        }
        _reader->seek(now_tem);
    }
    _isPushPause = false;
    _pause_time = now_tem;
    // _clock->jump(now_tem);
    Resume();
}

void PlayerImpl::Rewind(int interval_tiem) {
    Pause();
    _isPushPause = true;
    rcutils_time_point_value_t new_start =
        _pause_time - std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(interval_tiem)).count();
    int64_t new_point_1 = new_start > _starting_time ? new_start : _starting_time;
    int64_t new_point_2 = new_start > _first_message_time ? new_start : _first_message_time;
    {
        std::lock_guard<std::mutex> lk(_reader_mutex);
        {
            //清空缓存队列
            std::lock_guard<std::mutex> lk(_messageListMutex);
            std::queue<BagMessage*> emptyQueue;
            std::swap(_messageList, emptyQueue);
            _isSrorageEmpty = false;
        }
        _reader->seek(new_point_1);
    }
    _isPushPause = false;
    _pause_time = new_point_2;
    Resume();
}

void PlayerImpl::PublishH265Data() {
    ProcessUtility::SetThreadName("play_pub_h265");
    while (!_isStop) {
        // check pause or not.
        while (_isPopPause) {
            if (_isStop) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        BagMessage* bagMessage = nullptr;
        {
            std::unique_lock<std::mutex> lck(_h265Mutex);
            if (_h265MessageList.size() > 0) {
                bagMessage = _h265MessageList.front();
                _h265MessageList.pop();
            }
        }
        if (bagMessage) {
            auto currentTime =
                std::chrono::time_point_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now());
            uint64_t play_time = currentTime.time_since_epoch().count() - _clock->now();
            _h265_play->Play(play_time, bagMessage);
            delete bagMessage;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
}

bool PlayerImpl::PrepareSomeipPublishers(const PlayerOptions& playerOptions) {
    _is_someip = false;
    // std::string lidar_conf_path = playerOptions.lidar_conf_path;
    // struct stat buffer;
    // if (stat(lidar_conf_path.c_str(), &buffer) == 0) {
    //     // todo: 判断是否创建someip publisher 成功
    //     BAG_LOG_INFO << "create someip player publisher.";
    //     _uptr_someip_publisher = std::make_unique<SomeipPlayerPublisher>(lidar_conf_path);
    //     _is_someip = true;
    //     BAG_LOG_DEBUG << "create SomeipPlayerPublisher success.";
    // } else {
    //     BAG_LOG_WARN << "someip config file not exsit.";
    // }

    return _is_someip;
}

bool PlayerImpl::read_Attachment(const std::string file_name, std::string& data) {
    auto attachmentPtr = _reader->read_Attachment(file_name);
    if (attachmentPtr == nullptr) {
        return false;
    }
    // printf("attachmentPtr->datasize : %lu, data : %s\n",attachmentPtr->data.size(), attachmentPtr->data.c_str());
    data = attachmentPtr->data;
    return true;
}
}  // namespace bag
}  //namespace netaos
}  //namespace hozon
