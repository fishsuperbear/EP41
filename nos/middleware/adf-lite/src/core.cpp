#include <algorithm>
#include "adf-lite/include/core.h"
#include "adf-lite/include/adf_lite_internal_logger.h"
#include "adf-lite/include/dbg_info.h"
#include "config_param.h"
#include "adf/include/log.h"
#include "em/include/exec_client_impl_zmq.h"
#include "adf/include/node_proto_register.h"
#include "adf-lite/include/phm_client_instance.h"

namespace hozon {
namespace netaos {
namespace adf_lite {
using namespace hozon::netaos::cfg;
using namespace hozon::netaos::em;
using namespace hozon::netaos::phm;

Core::Core() {

}

Core::~Core() {

}

int32_t Core::Start(const std::string& top_config_file) {
    DO_OR_ERROR_EARLY(_config.Parse(top_config_file), "Fail to parse top file " << top_config_file);

    DO_OR_ERROR_EARLY(InitLogger(), "Fail to init logger.");

    ADF_INTERNAL_LOG_INFO << " ======================= Start ===================== .";
    std::shared_ptr<ExecClientImplZmq> _exec_client = std::make_shared<ExecClientImplZmq>();
    if (_exec_client) {
        _exec_client->ReportState(hozon::netaos::em::ExecutionState::kRunning);
    }

    PhmClientInstance::getInstance()->Init();

    DbgInfo::GetInstance().AddMgrMap(_executor_mgr_map);

    CfgResultCode initres = ConfigParam::Instance()->Init(1000);
    if (initres != CONFIG_OK) {
        ADF_INTERNAL_LOG_WARN << "fail to init configserver errorcode." << initres;
    }
    InitScheduler();

    int executor_count = _config.executors.size();
    std::vector<std::pair<uint32_t, std::string>> order_start_executor;
    std::vector<std::string> parallel_start_executor;
    for (int i = 0; i < executor_count; i++) {
        if (_config.executors[i].order < 100) {
            order_start_executor.emplace_back(_config.executors[i].order, _config.executors[i].config_file);
        } else {
            parallel_start_executor.emplace_back(_config.executors[i].config_file);
        }
        StartInitConfig(_config.executors[i].config_file);
    }
    std::stable_sort(order_start_executor.begin(), order_start_executor.end(),
        [](std::pair<uint32_t, std::string> ele1, std::pair<uint32_t, std::string> ele2){
            return ele1.first < ele2.first;
            });
    for (int i = 0; i < static_cast<int>(order_start_executor.size()); i++) {
        ADF_INTERNAL_LOG_DEBUG << "Start executor by order No." << i + 1 << " " << order_start_executor[i].second << ", order number = " << order_start_executor[i].first;
        StartExecutor(order_start_executor[i].second);
    }

    for (int i = 0; i < static_cast<int>(parallel_start_executor.size()); i++) {
        ADF_INTERNAL_LOG_DEBUG << "Start executor by parallel No." << i + 1 << " " << parallel_start_executor[i];
        _executor_thr.emplace_back(std::make_shared<std::thread>(std::bind(&hozon::netaos::adf_lite::Core::StartExecutor, *this, parallel_start_executor[i])));
    }

    for (int i = 0; i < static_cast<int>(_executor_thr.size()); i++) {
        _executor_thr[i]->join();
    }
    ADF_INTERNAL_LOG_INFO << "All Executor thread has been started";
    return 0;
}

void Core::StartInitConfig(const std::string& config_file) {
    if (config_file == "") {
        return;
    }
    std::shared_ptr<ExecutorMgr> mgr(new ExecutorMgr);

    if (!mgr) {
        ADF_INTERNAL_LOG_ERROR << config_file << " Fail";
    }

    if (mgr->InitConfig(config_file) != 0) {
        ADF_INTERNAL_LOG_ERROR << "Fail to InitConfig in " << config_file;
        SendFault_t fault(4900, 1, 1);
        int32_t result = PhmClientInstance::getInstance()->ReportFault(fault);
        if (result < 0) {
            ADF_INTERNAL_LOG_ERROR << "Fail to InitConfig ReportFault failed. failedCode: " << result;
        }
        return;
    }
    _executor_mgr_map[config_file] = mgr;
    ADF_INTERNAL_LOG_INFO << "Success to InitConfig in " << config_file;
}

int32_t Core::StartExecutor(const std::string& config_file) {

    ADF_INTERNAL_LOG_INFO << config_file << " Init";
    if (_executor_mgr_map[config_file]) {
        if (_executor_mgr_map[config_file]->Init()) {
            ADF_INTERNAL_LOG_ERROR << "Stop executor because Fail to init executor in " << config_file;
            _executor_mgr_map[config_file]->PreStop();
            _executor_mgr_map[config_file]->Stop();
            _executor_mgr_map.erase(config_file);
            SendFault_t fault(4900, 2, 1);
            int32_t result = PhmClientInstance::getInstance()->ReportFault(fault);
            if (result < 0) {
                ADF_INTERNAL_LOG_ERROR << "Fail to init executor ReportFault failed. failedCode: " << result;
            }
            return -1;
        }
        _executor_mgr_map[config_file]->StartProcess();
    }
    ADF_INTERNAL_LOG_INFO << "Start to process triggers: " << config_file;
    return 0;
}

void Core::Stop() {
    for (auto& executor: _executor_mgr_map) {
        auto& mgr = executor.second;
        if(mgr) {
            mgr->PreStop();
        }
    }

    for (auto& executor: _executor_mgr_map) {
        auto& mgr = executor.second;
        if(mgr) {
            mgr->Stop();
        }
    }
    CfgResultCode initres = ConfigParam::Instance()->DeInit();
        if (initres != CONFIG_OK) {
        ADF_INTERNAL_LOG_WARN << "fail to deinit configserver errorcode." << initres;
    }

    PhmClientInstance::getInstance()->DeInit();

    std::shared_ptr<ExecClientImplZmq> _exec_client = std::make_shared<ExecClientImplZmq>();
    if (_exec_client) {
        _exec_client->ReportState(hozon::netaos::em::ExecutionState::kTerminating);
    }
}

int32_t Core::InitLogger() {
    hozon::netaos::log::InitLogging(
            _config.log.name,
            _config.log.description,
            static_cast<hozon::netaos::log::LogLevel>(_config.log.level),
            static_cast<uint32_t>(_config.log.mode),
            _config.log.file,
            10,
            10 * 1024 * 1024);

    AdfInternalLogger::GetInstance()._logger.Init("ADFL", static_cast<hozon::netaos::adf_lite::LogLevel>(_config.log.adfl_level));

    NodeLogger::GetInstance().CreateLogger(
            _config.log.name,
            _config.log.description,
            static_cast<hozon::netaos::log::LogLevel>(_config.log.level)
        );
    return 0;
}

int32_t Core::InitScheduler() {
    sched_param param;
    if (!_config.schedule.HasValue()) {
        return 0;
    }

    if (_config.schedule.Value().policy == SCHED_OTHER) {
        param.sched_priority = 0;
    }
    else if ((_config.schedule.Value().policy == SCHED_FIFO)
        || (_config.schedule.Value().policy == SCHED_RR)) {
        param.sched_priority = _config.schedule.Value().priority;
    }
    else {
        ADF_INTERNAL_LOG_ERROR << "Unsupported scheduler.";
        return -1;
    }

    DO_OR_ERROR(sched_setscheduler(0, _config.schedule.Value().policy, &param), "Fail to set scheduler.");
    DO_OR_ERROR(sched_setaffinity(0, sizeof(_config.schedule.Value().affinity), &(_config.schedule.Value().affinity)), "Fail to set cpu affinity.");

    return 0;
}

}
}
}
