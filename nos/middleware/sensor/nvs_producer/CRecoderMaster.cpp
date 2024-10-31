
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <memory>

#include "CChannel.hpp"
#include "CUtils.hpp"
#include "CVirtualCamProducerChannel.hpp"
#include "cm/include/method.h"
#include "idl/generated/sensor_reattachPubSubTypes.h"
#include "nv_replayer_api.h"
#include "nvscibuf.h"
#include "nvscistream.h"
#include "nvscisync.h"

#ifndef MIDDLEWARE_SENSOR_NVS_PRODUCER_CRECODERMASTER_HPP_
#define MIDDLEWARE_SENSOR_NVS_PRODUCER_CRECODERMASTER_HPP_

using namespace nvsipl;

using hozon::netaos::codec::PicInfo;
using hozon::netaos::codec::PicInfos;

/** CMaster class */
class NvReplayer::CRecoderMaster {
   public:
    SIPLStatus CreateProducerChannels(const PicInfos& pic_infos) {

        pic_infos_ = pic_infos;
        auto status = Setup(1);
        CHK_STATUS_AND_RETURN(status, "CreateProducerChannels Setup");

        for (auto& info : pic_infos) {
            if (info.second.height) {
                vup_profilers_[info.second.sid] = unique_ptr<CProfiler>(new CProfiler());
                CHK_PTR_AND_RETURN(vup_profilers_[info.second.sid], "Profiler creation");
                auto status = RegisterSource(info.second.sid, vup_profilers_[info.second.sid].get());
                CHK_STATUS_AND_RETURN(status, "CreateProducerChannels RegisterSource");
            }
        }
        status = InitStream();
        CHK_STATUS_AND_RETURN(status, "CreateProducerChannels InitStream");

        // status = AsyncInitStreams();
        // CHK_STATUS_AND_RETURN(status, "CreateProducerChannels AsyncInitStreams");

        status = StartStream();
        CHK_STATUS_AND_RETURN(status, "CreateProducerChannels StartStream");

        reattach_serv_.Start(0, "sensor_reattach");

        // check_heartbeat_thread_.reset(new thread(&CRecoderMaster::CheckHeartbeatInfo, this));

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus Post(uint8_t sid, InputBufPtr buf) {
        SIPLStatus ret = NVSIPL_STATUS_ERROR;
        if (channels_[sid].get()) {
            CVirtualCamProducerChannel* ch = dynamic_cast<CVirtualCamProducerChannel*>(channels_[sid].get());
            ret = ch->Post(buf);
        }
        return ret;
    }

    SIPLStatus StopStream() {
        main_exit_ = true;
        if (reattach_serv_.Stop() < 0) {
            LOG_ERR("reattach_serv_.Stop() error\n");
        }
        if (check_heartbeat_thread_ != nullptr) {
            check_heartbeat_thread_->join();
            check_heartbeat_thread_.reset(0);
        }
        // need to release other nvsci resources before closing modules.
        for (auto i = 0U; i < MAX_NUM_SENSORS; i++) {
            if (nullptr != channels_[i]) {
                channels_[i]->Stop();
                channels_[i]->Deinit();
                channels_[i].reset();
            }
        }
        LOG_DBG("CMaster release.\n");
        if (scibuf_module_ != nullptr) {
            NvSciBufModuleClose(scibuf_module_);
        }

        if (sync_module_ != nullptr) {
            NvSciSyncModuleClose(sync_module_);
        }

        NvSciIpcDeinit();

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus AttachConsumer(uint32_t uSensor, uint32_t index) {
        if (nullptr != channels_[uSensor]) {
            CVirtualCamProducerChannel* pChannel = dynamic_cast<CVirtualCamProducerChannel*>(channels_[uSensor].get());
            pChannel->attach(index);
        }
        return NVSIPL_STATUS_OK;
    }

    SIPLStatus DetachConsumer(uint32_t uSensor, uint32_t index) {
        if (nullptr != channels_[uSensor]) {
            CVirtualCamProducerChannel* pChannel = dynamic_cast<CVirtualCamProducerChannel*>(channels_[uSensor].get());
            pChannel->detach(index);
        }
        return NVSIPL_STATUS_OK;
    }

    void UpdateHeartbeatInfo(uint32_t sid, uint32_t index) {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
        heartbeat_infos_[index][sid] = duration.count();
    }

   protected:
    SIPLStatus Setup(uint8_t multi_num) {
        consumer_num_ = multi_num;

        auto sciErr = NvSciBufModuleOpen(&scibuf_module_);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufModuleOpen");

        sciErr = NvSciSyncModuleOpen(&sync_module_);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncModuleOpen");

        sciErr = NvSciIpcInit();
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciIpcInit");
        LOG_ERR("NvSciIpcInit ok!\n");

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus StartStream(void) {
        for (auto i = 0U; i < MAX_NUM_SENSORS; i++) {
            if (nullptr != channels_[i]) {
                channels_[i]->Start();
            }
        }

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus RegisterSource(uint32_t sid, CProfiler* profiler) {
        LOG_ERR("CRecoderMaster: RegisterSource. sid=%d\n", sid);

        if (sid >= MAX_NUM_SENSORS) {
            LOG_ERR("%s: Invalid sensor id: %u\n", __func__, sid);
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }
        sensor_infos_[sid].reset(new SensorInfo);
        sensor_infos_[sid]->id = sid;

        channels_[sid] = CreateChannel(&pic_infos_[sid], sensor_infos_[sid].get());
        CHK_PTR_AND_RETURN(channels_[sid], "Master CreateChannel");

        channels_[sid]->Init();

        auto status = channels_[sid]->CreateBlocks(profiler);
        CHK_STATUS_AND_RETURN(status, "Master CreateBlocks");

        return NVSIPL_STATUS_OK;
    }

    void CreateStream(const uint32_t index) {
        string tname = "CreateStream" + std::to_string(index);
        pthread_setname_np(pthread_self(), tname.c_str());
        auto status = channels_[index]->Connect();
        is_init_stream_done_[index] = false;
        if (status == NVSIPL_STATUS_OK) {
            status = channels_[index]->InitBlocks();
            if (status == NVSIPL_STATUS_OK) {
                status = channels_[index]->Reconcile();
                if (status == NVSIPL_STATUS_OK)
                    is_init_stream_done_[index] = true;
            }
        }
        return;
    }

    SIPLStatus AsyncInitStreams(void) {
        LOG_DBG("CMaster: InitStream.\n");

        auto ret = NVSIPL_STATUS_OK;
        for (auto i = 0U; i < MAX_NUM_SENSORS; i++) {
            if (nullptr != channels_[i]) {
                init_threads_[i].reset(new std::thread(&CRecoderMaster::CreateStream, this, i));
                if (init_threads_[i] == nullptr) {
                    LOG_ERR("Failed to create InitStream thread\n");
                    ret = NVSIPL_STATUS_ERROR;
                    break;
                }
            }
        }
        for (auto i = 0U; i < MAX_NUM_SENSORS; i++) {
            if (init_threads_[i] != nullptr) {
                init_threads_[i]->join();
                init_threads_[i].reset();
                if (!is_init_stream_done_[i]) {
                    LOG_ERR("Init sersorID:%d stream channel fail!", i);
                    ret = NVSIPL_STATUS_ERROR;
                }
            }
        }
        return ret;
    }

    SIPLStatus InitStream(void) {
        LOG_DBG("CMaster: InitStream.\n");

        for (auto i = 0U; i < MAX_NUM_SENSORS; i++) {
            if (nullptr != channels_[i]) {
                auto status = channels_[i]->Connect();
                CHK_STATUS_AND_RETURN(status, "CMaster: Channel connect.");

                status = channels_[i]->InitBlocks();
                CHK_STATUS_AND_RETURN(status, "InitBlocks");

                status = channels_[i]->Reconcile();
                CHK_STATUS_AND_RETURN(status, "Channel Reconcile");
            }
        }

        return NVSIPL_STATUS_OK;
    }

    void CheckHeartbeatInfo() {
        while (!main_exit_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            auto now = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
            for (int i = 0; i < NUM_IPC_CONSUMERS; ++i) {
                for (int j = 0; j < MAX_NUM_SENSORS; ++j) {
                    if (heartbeat_infos_[i][j] != 0 && (duration.count() - heartbeat_infos_[i][j] > 300)) {
                        LOG_ERR("consumer lost connection, DetachConsumer! sid=%d, index=%d\n", i, j);
                        auto ret = DetachConsumer(i, j);
                        if (ret != NVSIPL_STATUS_OK) {
                            LOG_ERR("DetachConsumer fail! sid=%d, index=%d\n", i, j);
                        }
                    }
                }
            }
        }
    }

   private:
    std::unique_ptr<CChannel> CreateChannel(PicInfo* pic_info, SensorInfo* sensor_info) {
        // create channel by recorder file infos.
        return std::unique_ptr<CVirtualCamProducerChannel>(
            new CVirtualCamProducerChannel(scibuf_module_, sync_module_, pic_info, sensor_info));
    }

    std::shared_ptr<sensor_reattachPubSubType> req_data_type = std::make_shared<sensor_reattachPubSubType>();
    std::shared_ptr<sensor_reattach_respPubSubType> resp_data_type = std::make_shared<sensor_reattach_respPubSubType>();

    class ReattachServer : public hozon::netaos::cm::Server<sensor_reattach, sensor_reattach_resp> {
       public:
        ReattachServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data,
                       std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data)
            : Server(req_data, resp_data) {}

        int32_t Process(const std::shared_ptr<sensor_reattach> req, std::shared_ptr<sensor_reattach_resp> resp) {
            LOG_MSG("[ReattachServer]recv reattach request:%d,%d,%d.\n", req->isattach(), req->sensor_id(),
                    req->index());
            // if (req->isalive()) {
            //     This()->UpdateHeartbeatInfo(req->sensor_id(), req->index());
            //     return 0;
            // }

            if (req->isattach()) {
                This()->AttachConsumer(req->sensor_id(), req->index());
            } else {
                This()->DetachConsumer(req->sensor_id(), req->index());
            }

            return 0;
        }

        using BaseClass = NvReplayer::CRecoderMaster;

        BaseClass* This() {
            return reinterpret_cast<BaseClass*>(reinterpret_cast<char*>(this) - offsetof(BaseClass, reattach_serv_));
        }
    } reattach_serv_{req_data_type, resp_data_type};

    std::atomic_bool main_exit_{false};
    NvSciSyncModule sync_module_{nullptr};
    NvSciBufModule scibuf_module_{nullptr};
    std::unique_ptr<CChannel> channels_[MAX_NUM_SENSORS]{nullptr};
    PicInfos pic_infos_;
    unique_ptr<CProfiler> vup_profilers_[MAX_NUM_SENSORS]{nullptr};
    unique_ptr<SensorInfo> sensor_infos_[MAX_NUM_SENSORS]{nullptr};
    uint8_t consumer_num_ = 1;
    bool is_init_stream_done_[MAX_NUM_SENSORS]{false};
    unique_ptr<std::thread> init_threads_[MAX_NUM_SENSORS]{nullptr};
    uint64_t heartbeat_infos_[NUM_IPC_CONSUMERS][MAX_NUM_SENSORS] = {0};
    unique_ptr<std::thread> check_heartbeat_thread_ = nullptr;
};

bool NvReplayer::CreateProducerChannels(const PicInfos& pic_infos) {
    impl_->CreateProducerChannels(pic_infos);
    return true;
}

// bool NvReplayer::Post(uint8_t sid, const InputBuf& buf) {
//     impl_->Post(sid, buf);
//     return true;
// }

bool NvReplayer::Post(uint8_t sid, InputBufPtr buf) {
    impl_->Post(sid, buf);
    return true;
}

bool NvReplayer::CloseAllChannels() {
    impl_->StopStream();
    return true;
}

NvReplayer::NvReplayer() : impl_(std::make_unique<CRecoderMaster>()) {
    CLogger::GetInstance().SetLogLevel(LEVEL_ERR);
    CLogger::GetInstance().InitLogger();
}

NvReplayer::~NvReplayer() = default;

#endif  // MIDDLEWARE_SENSOR_NVS_PRODUCER_CRECODERMASTER_HPP_
