#pragma once
#include <chrono>
#include <fstream>
#include <memory>
#include <set>
#include <unordered_map>
#include "cm/include/skeleton.h"
// #include "cm/include/skeleton.h"
#include "bag_message.hpp"
#include "middleware/codec/include/atomic_queue.h"
#include "middleware/codec/include/decoder_factory.h"
#include "proto/soc/sensor_image.pb.h"
// dds
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/dds/publisher/DataWriterListener.hpp>
#include <fastdds/rtps/transport/shared_mem/SharedMemTransportDescriptor.h>
#include "cm/include/cm_config.h"
#include "fastdds/dds/publisher/Publisher.hpp"
#include "fastdds/rtps/transport/UDPv4TransportDescriptor.h"
#include "idl/generated/proto_methodTypeObject.h"
#include "idl/generated/zerocopy_imagePubSubTypes.h"
#include "idl/generated/zerocopy_imageTypeObject.h"
#include "topic_manager.hpp"

using namespace eprosima::fastdds::dds;

namespace hozon {
namespace netaos {
namespace bag {

const std::set<std::string> h265_set{"/soc/encoded_camera_0",  "/soc/encoded_camera_1", "/soc/encoded_camera_10",
                                     "/soc/encoded_camera_11", "/soc/encoded_camera_2", "/soc/encoded_camera_4",
                                     "/soc/encoded_camera_5",  "/soc/encoded_camera_6", "/soc/encoded_camera_7",
                                     "/soc/encoded_camera_8",  "/soc/encoded_camera_9"};

class H265PlayHandler {
   public:
    H265PlayHandler(const std::string& cam_count,
                    std::shared_ptr<hozon::netaos::data_tool_common::TopicManager> topic_manager);
    ~H265PlayHandler();
    void Play(int64_t diff_time, BagMessage* h265_message);
    bool IsReady();
    void Stop();
    void SetPause(bool is_pause);

   protected:
    typedef struct {
        uint64_t post_time;
        uint8_t frame_type;
        std::shared_ptr<hozon::soc::CompressedImage> img;
    } DecodeTask;

    using DecodeTaskPtr = std::shared_ptr<DecodeTask>;

    typedef struct {
        uint8_t sid;
        std::unique_ptr<std::thread> work_thread;
        using AtomicQueue = atomic_queue::AtomicQueue2<DecodeTaskPtr, 1, false, false, false, true>;
        AtomicQueue queue;
        std::unique_ptr<codec::Decoder> codec;
        std::string encode;
        eprosima::fastdds::dds::DataWriter* writer;
    } Worker;

    struct DdsCtx {};

    uint8_t GetSid(const std::string& topic);
    bool IsSkipTopic(uint8_t sid);
    void WorkThread(Worker& worker);
    bool CreateCudaDecoder(const codec::PicInfos& pic_info);
    bool CreateCpuDecoder(const codec::PicInfos& pic_info);
    std::unique_ptr<codec::Decoder> codec_;
    std::map<uint32_t, bool> sensor_1st_iframe_flags_;
    uint8_t camera_count_ = 0;
    std::unordered_map<uint8_t, std::unique_ptr<Worker>> workers_;

    bool is_cpu_ = true;
    std::unique_ptr<std::ofstream> files_[16] = {0};

    std::shared_ptr<DomainParticipant> participant_ = nullptr;
    std::shared_ptr<Publisher> publisher_ = nullptr;
    std::vector<Topic*> topiclist_;
    bool stopped_ = false;
    std::atomic_bool is_pause_{false};

    std::shared_ptr<hozon::netaos::data_tool_common::TopicManager> topic_manager_;
    std::string target_platform_ = "";
};

}  // namespace bag
}  // namespace netaos
}  // namespace hozon
