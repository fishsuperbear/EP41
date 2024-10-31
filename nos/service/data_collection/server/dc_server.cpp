//
// Created by cheng on 23-6-20.
//

#include <signal.h>
#include <semaphore.h>
#include <unistd.h>
#include <condition_variable>
#include <semaphore.h>
#include <iostream>  // for_debug
#include <utility>

#include "cm/include/method.h"
#include "config_param.h"
#include "collection/include/collection_manager.h"
#include "idl/generated/data_collection_info.h"
#include "idl/generated/data_collection_infoPubSubTypes.h"
#include "manager/include/cfg_calibstatus.h"
#include "data_tools/common/util/include/data_tools_logger.hpp"
#include "log/include/default_logger.h"
#include "pipeline/include/pipeline_manager.h"
#include "recorder.h"
#include "utils/include/dc_logger.hpp"
#include "em/include/exec_client.h"
#include "tsp_comm.h"

using namespace hozon::netaos::cm;
using namespace hozon::netaos::bag;
using namespace hozon::netaos::dc;
using namespace hozon::netaos::em;
using namespace std;

sem_t g_semaphore;
//std::condition_variable g_waitStop;
//std::atomic_bool g_stop;
//std::mutex g_mtx;

class DcServer : public Server<triggerInfo, triggerResult> {
   public:
    DcServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_topic_type, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_topic_type)
        : Server<triggerInfo, triggerResult>(std::move(req_topic_type), std::move(resp_topic_type)) {}
    ~DcServer() {
    }

    int32_t Process(const std::shared_ptr<triggerInfo> req, std::shared_ptr<triggerResult> resp) override {
        cout << "receive client Request" << endl;
        PipelineManager::getInstance()->trigger(req, resp);
        resp->msg("OK");
        resp->retCode(0);
        cout << "--- finish\n";
        return 0;
    };

   private:
    //  CollectionManager *cm_;
};

class DcUploadServer : public Server<triggerUploadInfo, triggerResult> {
   public:
    DcUploadServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_topic_type, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_topic_type)
        : Server<triggerUploadInfo, triggerResult>(std::move(req_topic_type), std::move(resp_topic_type)) {}
    ~DcUploadServer() {
    }

    int32_t Process(const std::shared_ptr<triggerUploadInfo> req, std::shared_ptr<triggerResult> resp) override {
        cout << "receive client Upload Request" << endl;
        PipelineManager::getInstance()->triggerUpload(req, resp);
        resp->msg("OK");
        resp->retCode(0);
        cout << "--- upload finish\n";
        return 0;
    };

   private:
};

// 定义信号处理程序
void signalHandler(int signum) {
    // 终止程序
    cout << "\n\nstop start" << endl;
    DC_SERVER_LOG_WARN << "receive stop message";
    sem_post(&g_semaphore);
//    g_stop.store(true, std::memory_order_release);
//    g_waitStop.notify_all();
}

int main(int argc, char* argv[]) {
    sem_init(&g_semaphore, 0, 0);
    if (PathUtils::isFileExist(PathUtils::debugModeFilePath)) {
        hozon::netaos::log::InitLogging("DC", "NETAOS DC", hozon::netaos::log::LogLevel::kDebug,
                                        hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE, "/opt/usr/log/soc_log/", 10, (20*1024*1024),true);
    } else {
        hozon::netaos::log::InitLogging("DC", "NETAOS DC", hozon::netaos::log::LogLevel::kInfo,
                                        hozon::netaos::log::HZ_LOG2FILE, "/opt/usr/log/soc_log/", 10, (20 * 1024 * 1024),true);
    }
    std::shared_ptr<ExecClient> execli = std::make_shared<ExecClient>();
    for (int i = 0; i < 4; i++) {
        int32_t ret = execli->ReportState(ExecutionState::kRunning);
        if (ret) {
            DC_SERVER_LOG_WARN << "data_collection report running failed";
        } else {
            break;
        }
    }
    DC_OPER_LOG_INFO<<"###################DC start###########################";
    CalibStatusReceive csr;
    csr.Init();
    hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetCommonLogger()->SetLogLevel(hozon::netaos::log::LogLevel::kError);
    hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetBagLogger()->SetLogLevel(hozon::netaos::log::LogLevel::kError);
    hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetTopicLogger()->SetLogLevel(hozon::netaos::log::LogLevel::kError);
    std::filesystem::path needCleanPath(FLOW_DEBUG_DATA_PATH);
    if (FLOW_DEBUG_DATA_PATH.size()>10) {
        PathUtils::removeFilesInFolder(FLOW_DEBUG_DATA_PATH);
    }
//    g_stop.store(false, std::memory_order_release);
    DcServer dcServer(make_shared<triggerInfoPubSubType>(), std::make_shared<triggerResultPubSubType>());
    DcUploadServer dcUploadServer(make_shared<triggerUploadInfoPubSubType>(), std::make_shared<triggerResultPubSubType>());
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    auto cfgMgr = hozon::netaos::cfg::ConfigParam::Instance();
    cfgMgr->Init(5000);
    auto& tsp = hozon::netaos::https::TspComm::GetInstance();
    tsp.Init();

    PipelineManager::getInstance()->start();
    dcServer.Start(0, "serviceDataCollection");
    dcUploadServer.Start(0, "serviceDataCollectionUpload");


//    unique_lock<mutex> ulock(g_mtx);
//    g_waitStop.wait(ulock,[]{ return g_stop.load(std::memory_order::memory_order_acquire);});
    DC_OPER_LOG_INFO << "data_collection inited finish. wait signal.";
    sem_wait(&g_semaphore);
//    TimeUtils::sleep(7000);
    DC_OPER_LOG_INFO << "####################DC stop###############################";
    std::thread emReport([&execli] {
        auto ret2 = execli->ReportState(ExecutionState::kTerminating);
        if (ret2) {
            DC_SERVER_LOG_ERROR << "data_collection report terminating failed";
        }
    });
    std::thread dcServerStop([&dcServer] {
        dcServer.Stop();
        DC_OPER_LOG_INFO << "DcStop:dcServer stoped.";
    });
    std::thread dcUploadStop([&dcUploadServer] {
        dcUploadServer.Stop();
        DC_OPER_LOG_INFO << "DcStop:dcUploadServer stoped.";
    });
    std::thread thirdDeInit([&cfgMgr, &tsp, &csr] {
        csr.DeInit();
        DC_OPER_LOG_INFO << "DcStop:cfgMgr stoped.";
        cfgMgr->DeInit();
        DC_OPER_LOG_INFO << "DcStop:cfgMgr stoped.";
        tsp.Deinit();
        DC_OPER_LOG_INFO << "DcStop:tsp stoped.";
    });
    PipelineManager::getInstance()->stop();
    DC_OPER_LOG_INFO << "DcStop:PipelineManager stoped.";
    if (emReport.joinable()) {
        emReport.join();
    }
    if (dcServerStop.joinable()) {
        dcServerStop.join();
    }
    if (dcUploadStop.joinable()) {
        dcUploadStop.join();
    }
    if (thirdDeInit.joinable()) {
        thirdDeInit.join();
    }
    DC_OPER_LOG_INFO << "####################DC exit###############################";
    return 0;
}

//void testMainFunc() {
//
//    signal(SIGINT, signalHandler);
//    const int recNum = 2;
//    const int recTimeS = 15;
//    const int offsetS = 1;
//    Recorder recorder[recNum];
//    Recorder *recP = recorder;
//    std::thread recThread[recNum * 2];
//    RecordOptions rops;
//    rops.max_bagfile_size = 2,
//    rops.max_bagfile_duration = 2,
//    rops.record_all = true,
//    rops.max_files = 4;
//    for (int i = 0; i < recNum; i++) {
//        recThread[2 * i] = std::thread([i, recP, &rops, offsetS] {
//            std::this_thread::sleep_for(std::chrono::seconds(offsetS * i));
//            recP[i].Start(rops);
//        });
//        recThread[2 * i + 1] = std::thread([i, recP, recTimeS] {
//            std::this_thread::sleep_for(std::chrono::seconds(recTimeS));
//            recP[i].Stop();
//        });
//    }
//    for (auto &th : recThread) {
//        th.join();
//    }
//
//    cout << "stop end" << endl;
//}
