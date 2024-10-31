#include "h265_play_handler.h"
#include <fastdds/dds/publisher/DataWriter.hpp>
#include "bag_data_pubsubtype.h"
#include "cm/include/cm_config.h"
#include "fastdds/dds/publisher/Publisher.hpp"
#include "idl/generated/cm_protobuf.h"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "idl/generated/cm_protobufTypeObject.h"
#include "topic_manager.hpp"
#include "util/include/data_tools_logger.hpp"
#include "proto/soc/sensor_image.pb.h"

namespace hozon {
namespace netaos {
namespace bag {

static codec::PicInfo g_sensor_info[] = {
    {2160, 3840, 0, 8},   // 0
    {2160, 3840, 1, 8},   // 1
    {1080, 1920, 2, 8},   // 2
    {0, 0, 0, 0},         // 3
    {1080, 1920, 4, 8},   // 4
    {1080, 1920, 5, 8},   // 5
    {1080, 1920, 6, 8},   // 6
    {1080, 1920, 7, 8},   // 7
    {1080, 1920, 8, 8},   // 8
    {1080, 1920, 9, 8},   // 9
    {1080, 1920, 10, 8},  // 10
    {1080, 1920, 11, 8},  // 11
    {0, 0, 0, 0},         // 12
    {0, 0, 0, 0},         // 13
    {0, 0, 0, 0},         // 14
    {0, 0, 0, 0},         // 15
};

static int8_t g_cam_info[3][16] = {
    {0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, -1, -1, -1, -1, -1},        // 11V
    {0, 1, 2, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},      // 7V
    {8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}  // 4V
};

H265PlayHandler::H265PlayHandler(const std::string& cam_count,
                                 std::shared_ptr<hozon::netaos::data_tool_common::TopicManager> topic_manager) {
    topic_manager_ = topic_manager;
    if (TARGET_PLATFORM == "orin") {
        target_platform_ = "orin";
    } else if (TARGET_PLATFORM == "x86_2004") {
        target_platform_ = "x86_2004";
    }
    stopped_ = false;

    auto idx = cam_count.find("c");
    if (idx != std::string::npos) {
        camera_count_ = std::atoi(cam_count.substr(idx + 1).data());
        is_cpu_ = true;
    } else {
        camera_count_ = std::atoi(cam_count.data());
        is_cpu_ = false;
    }

    codec::PicInfos infos;

    if (camera_count_ == 7) {
        for (auto i : g_cam_info[1]) {
            if (i != -1) {
                auto sinfo = g_sensor_info[i];
                infos[sinfo.sid] = sinfo;
            }
        }
    } else if (camera_count_ == 4) {
        for (auto i : g_cam_info[2]) {
            if (i != -1) {
                auto sinfo = g_sensor_info[i];
                infos[sinfo.sid] = sinfo;
            }
        }
    } else {
        for (auto i : g_cam_info[0]) {
            if (i != -1) {
                auto sinfo = g_sensor_info[i];
                infos[sinfo.sid] = sinfo;
            }
        }
    }

    if (target_platform_ == "orin") {
        codec_ = codec::DecoderFactory::Create(codec::kDeviceType_NvMedia_NvStream);
        codec_->Init(infos);
    } else if (target_platform_ == "x86_2004") {
        // Craete participant for subscribing normal topics.
        // auto participant_ = hozon::netaos::data_tool_common::TopicManager::GetInstance().GetParticipant(hozon::netaos::data_tool_common::kDdsDataType_Normal, true);
        // auto publisher_ = hozon::netaos::data_tool_common::TopicManager::GetInstance().GetPublisher(hozon::netaos::data_tool_common::kDdsDataType_Normal);

        participant_ = topic_manager_->GetParticipant(hozon::netaos::data_tool_common::kDdsDataType_CameraYuv, true);
        publisher_ = topic_manager_->GetPublisher(hozon::netaos::data_tool_common::kDdsDataType_CameraYuv);

        // REGISTER THE TYPE
        registerzerocopy_imageTypes();

        auto ztype_8m420 = std::make_shared<ZeroCopyImg8M420PubSubType>();
        eprosima::fastdds::dds::TypeSupport type1(ztype_8m420);
        type1.get()->auto_fill_type_information(true);
        type1.get()->auto_fill_type_object(false);
        type1.register_type(participant_.get());

        auto ztype_2m422 = std::make_shared<ZeroCopyImg2M422PubSubType>();
        eprosima::fastdds::dds::TypeSupport type2(ztype_2m422);
        type2.get()->auto_fill_type_information(true);
        type2.get()->auto_fill_type_object(false);
        type2.register_type(participant_.get());

        ////////////////////////////////////////////////////////////////////
        if (is_cpu_) {
            CreateCpuDecoder(infos);
        } else {
            if (!CreateCudaDecoder(infos)) {
                // clear workers_
                std::unordered_map<uint8_t, std::unique_ptr<Worker>> del{0};
                workers_.swap(del);
            }
        }

        // start all work thread.
        for (auto& i : workers_) {
            workers_[i.first]->work_thread.reset(
                new std::thread(&H265PlayHandler::WorkThread, this, std::ref(*workers_[i.first])));
        }
    }
}

bool H265PlayHandler::CreateCudaDecoder(const codec::PicInfos& pic_info) {
    codec::DecodeInitParam init_param{0};
    init_param.codec_type = codec::kCodecType_H265;

    for (auto& i : pic_info) {
        auto worker = std::make_unique<Worker>();
        auto codec = codec::DecoderFactory::Create(codec::kDeviceType_Cuda);
        std::string topic_type = "";
        if (i.first == 0 || i.first == 1) {
            init_param.yuv_type = codec::kYuvType_NV12;
            init_param.width = 3840;
            init_param.height = 2160;
            worker->encode = "NV12";
            topic_type = "ZeroCopyImg8M420";
        } else {
            init_param.yuv_type = codec::kYuvType_YUYV;
            init_param.width = 1920;
            init_param.height = 1080;
            worker->encode = "YUYV";
            topic_type = "ZeroCopyImg2M422";
        }
        init_param.sid = i.first;

        auto ret = codec->Init(init_param);
        if (ret == codec::kDeviceNotSupported) {
            return false;
        } else {
            is_cpu_ = false;
            // init cm
            std::string topic_name = "/soc/zerocopy/camera_" + std::to_string(i.first);
            /////////////////////////////////////////////////////////////////
            //REGISTER THE TYPE
            Topic* topic = participant_->create_topic(topic_name, topic_type, TOPIC_QOS_DEFAULT);

            if (topic == nullptr) {
                BAG_LOG_ERROR << "create_topic failed! topic:" << topic_name << " type:" << topic_type;
                return false;
            }
            topiclist_.push_back(topic);

            // CREATE THE WRITER
            DataWriterQos wqos =
                topic_manager_->GetWriterQos(hozon::netaos::data_tool_common::kDdsDataType_CameraYuv, topic_name);

            DataWriter* writer = publisher_->create_datawriter(topic, wqos);
            if (nullptr == writer) {
                BAG_LOG_ERROR << "create DataWriter failed! topic:" << topic_name << " type:" << topic_type;
            }

            worker->writer = writer;
            BAG_LOG_INFO << "ready to play topic:" << topic_name << " type:" << topic_type;
            ////////////////////////////////////////////////////////////////////////
            // init worker
            worker->codec = std::move(codec);
            worker->sid = i.first;
            workers_[i.first] = std::move(worker);
        }
    }
    printf("use cuda codec.\n");
    return true;
}

bool H265PlayHandler::CreateCpuDecoder(const codec::PicInfos& pic_info) {
    codec::DecodeInitParam init_param{0};
    init_param.codec_type = codec::kCodecType_H265;

    for (auto& i : pic_info) {
        auto worker = std::make_unique<Worker>();
        auto codec = codec::DecoderFactory::Create(codec::kDeviceType_Cpu);
        std::string topic_type = "";
        if (i.first == 0 || i.first == 1) {
            init_param.yuv_type = codec::kYuvType_NV12;
            init_param.width = 3840;
            init_param.height = 2160;
            worker->encode = "NV12";
            topic_type = "ZeroCopyImg8M420";
        } else {
            init_param.yuv_type = codec::kYuvType_YUYV;
            init_param.width = 1920;
            init_param.height = 1080;
            worker->encode = "YUYV";
            topic_type = "ZeroCopyImg2M422";
        }
        init_param.sid = i.first;

        auto ret = codec->Init(init_param);
        if (ret == codec::kEncodeSuccess) {
            // init cm
            std::string topic_name = "/soc/zerocopy/camera_" + std::to_string(i.first);
            /////////////////////////////////////////////////////////////////
            //REGISTER THE TYPE

            Topic* topic = participant_->create_topic(topic_name, topic_type, TOPIC_QOS_DEFAULT);

            if (topic == nullptr) {
                BAG_LOG_ERROR << "create_topic failed! topic:" << topic_name << " type:" << topic_type;
                return false;
            }
            topiclist_.push_back(topic);

            // CREATE THE WRITER
            DataWriterQos wqos =
                topic_manager_->GetWriterQos(hozon::netaos::data_tool_common::kDdsDataType_CameraYuv, topic_name);

            DataWriter* writer = publisher_->create_datawriter(topic, wqos);
            if (nullptr == writer) {
                BAG_LOG_ERROR << "create DataWriter failed! topic:" << topic_name << " type:" << topic_type;
            }

            worker->writer = writer;
            BAG_LOG_INFO << "ready to play topic:" << topic_name << " type:" << topic_type;
            ////////////////////////////////////////////////////////////////////////
            // init worker
            worker->codec = std::move(codec);
            worker->sid = i.first;
            workers_[i.first] = std::move(worker);
        }
    }
    printf("use cpu codec.\n");
    return true;
}

H265PlayHandler::~H265PlayHandler() {
    if (!stopped_) {
        Stop();
    }
}

void H265PlayHandler::Stop() {
    stopped_ = true;
    for (auto& worker : workers_) {
        if (worker.second->work_thread && worker.second->work_thread->joinable()) {
            worker.second->work_thread->join();
        }
    }

    for (auto& worker : workers_) {
        if (worker.second) {
            publisher_->delete_datawriter(worker.second->writer);
        }

        worker.second->work_thread = nullptr;
        worker.second->codec = nullptr;
        worker.second = nullptr;
    }
    workers_.clear();

    for (auto topic : topiclist_) {
        if (topic != nullptr) {
            participant_->delete_topic(topic);
        }
    }

    publisher_ = nullptr;
    participant_ = nullptr;
}

void H265PlayHandler::WorkThread(Worker& worker) {
    while (!stopped_) {
        DecodeTaskPtr task;
        if (!worker.queue.try_pop(task)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        void* buf = nullptr;
        auto ret = worker.writer->loan_sample(buf);

        if (ret != ReturnCode_t::RETCODE_OK) {
            BAG_LOG_ERROR << "loan_sample error  ret = " << ret();
            continue;
        }

        void* img = nullptr;
        auto header = static_cast<ZeroCopyImageHeader*>(buf);
        int32_t len = 0;
        uint8_t yuv_type = 0;

        if (worker.sid == 0 || worker.sid == 1) {
            auto s = static_cast<ZeroCopyImg8M420*>(buf);
            img = (void*)s->data().data();
            yuv_type = 0;
        } else {
            auto s = static_cast<ZeroCopyImg2M422*>(buf);
            img = (void*)s->data().data();
            yuv_type = 1;
        }

        header->height(worker.codec->GetHeight());
        header->width(worker.codec->GetWidth());
        // header->pushlish_timestamp(task->post_time);
        header->pushlish_timestamp(task->img->measurement_time() * 1000000000);
        header->sensor_timestamp(task->img->header().sensor_stamp().camera_stamp() * 1000000000);
        header->yuv_type(yuv_type);
        header->stride(worker.codec->GetStride());
        header->sid(worker.sid);
        header->frame_count(task->img->header().seq());

        worker.codec->Process(task->img->data(), img, &len);
        header->length(len);

        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());

        if ((uint64_t)duration.count() < task->post_time) {
            auto wait = std::chrono::nanoseconds(task->post_time) - duration;
            std::this_thread::sleep_for(wait);
        } else {
            auto wait = std::chrono::nanoseconds(task->post_time) - duration;
            BAG_LOG_ERROR << worker.sid << " | post_time before now, send directly!" << (int64_t)wait.count() / 1000000
                          << "ms";

            // worker.writer->discard_loan(buf);
            // continue;
        }

        if (worker.writer) {
            if (!worker.writer->write(buf)) {
                worker.writer->discard_loan(buf);
                BAG_LOG_INFO << "Write data to cm failed. Topic: ";
            } else {
                BAG_LOG_DEBUG << "write sucess topic = " << worker.writer->get_topic()->get_name();
            }
        }
    }
}

void H265PlayHandler::Play(int64_t diff_time, BagMessage* h265_message) {
    if (stopped_) {
        return;
    }

    if ("CmProtoBuf" == h265_message->type) {
        CmProtoBuf temp_cmProtoBuf;
        CmProtoBufPubSubType sub_type;
        sub_type.deserialize(h265_message->data.m_payload.get(), &temp_cmProtoBuf);
        auto prototype_msg = std::make_shared<hozon::soc::CompressedImage>();
        prototype_msg->ParseFromArray(temp_cmProtoBuf.str().data(), temp_cmProtoBuf.str().size());
        auto sid = GetSid(h265_message->topic);
        if (!sensor_1st_iframe_flags_[sid] && (prototype_msg->frame_type() != hozon::netaos::codec::kFrameType_I)) {
            // skip all p frames before 1st i-frame.
            return;
        } else {
            sensor_1st_iframe_flags_[sid] = true;
        }
        auto exposure_time = (uint64_t)(prototype_msg->measurement_time() * 1000000000);
        auto playtime = diff_time + exposure_time;

        if (!IsSkipTopic(sid)) {
            if (target_platform_ == "orin") {
                codec_->Process({playtime, exposure_time, (uint8_t)prototype_msg->frame_type(), sid},
                                prototype_msg->data());
            } else if (target_platform_ == "x86_2004") {
                auto task = std::make_shared<DecodeTask>();
                task->img = prototype_msg;
                task->post_time = playtime;

                auto worker = workers_.find(sid);
                if (worker == workers_.end()) {
                    BAG_LOG_ERROR << "work not find, sid=" << sid;
                } else {
                    // TODO(zax): instead of spinlock.
                    while (!stopped_ && !worker->second->queue.try_push(std::move(task))) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                }
            } else {
                BAG_LOG_ERROR << "unknown platform";
            }
        } else {
            BAG_LOG_INFO << "topic no need process. " << h265_message->topic;
        }
    } else {
        BAG_LOG_ERROR << "h265 data in type: " << h265_message->type << " is unsupported.";
    }
}

bool H265PlayHandler::IsReady() {

    return false;
}

uint8_t H265PlayHandler::GetSid(const std::string& topic) {
    uint8_t sid = 0;
    if (topic == "/soc/encoded_camera_0") {
        sid = 0;
    } else if (topic == "/soc/encoded_camera_1") {
        sid = 1;
    } else if (topic == "/soc/encoded_camera_2") {
        sid = 2;
    } else if (topic == "/soc/encoded_camera_3") {
        sid = 3;
    } else if (topic == "/soc/encoded_camera_4") {
        sid = 4;
    } else if (topic == "/soc/encoded_camera_5") {
        sid = 5;
    } else if (topic == "/soc/encoded_camera_6") {
        sid = 6;
    } else if (topic == "/soc/encoded_camera_7") {
        sid = 7;
    } else if (topic == "/soc/encoded_camera_8") {
        sid = 8;
    } else if (topic == "/soc/encoded_camera_9") {
        sid = 9;
    } else if (topic == "/soc/encoded_camera_10") {
        sid = 10;
    } else if (topic == "/soc/encoded_camera_11") {
        sid = 11;
    } else {
        BAG_LOG_ERROR << "error topic " << topic;
    }
    return sid;
}

bool H265PlayHandler::IsSkipTopic(uint8_t sid) {
    if (camera_count_ == 7) {
        for (auto i : g_cam_info[1]) {
            if (i == sid) {
                return false;
            }
        }
    } else if (camera_count_ == 4) {
        for (auto i : g_cam_info[2]) {
            if (i == sid) {
                return false;
            }
        }
    } else {
        for (auto i : g_cam_info[0]) {
            if (i == sid) {
                return false;
            }
        }
    }

    return true;
}

void H265PlayHandler::SetPause(bool is_pause) {
    is_pause_ = is_pause;
}

}  // namespace bag
}  // namespace netaos
}  // namespace hozon
