#include "phm_server/include/common/phm_server_logger.h"
#include "phm_server/include/common/phm_server_utils.h"
#include "phm_server/include/fault_manager/manager/phm_fault_task_handler.h"

namespace hozon {
namespace netaos {
namespace phm_server {

FaultTaskHandler* FaultTaskHandler::instance_ = nullptr;
std::mutex FaultTaskHandler::mtx_;
const std::string fault_thread_name = "fault_queue";

FaultTaskHandler*
FaultTaskHandler::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new FaultTaskHandler();
        }
    }

    return instance_;
}

FaultTaskHandler::FaultTaskHandler()
: stop_flag_(false)
{
}

void
FaultTaskHandler::Init()
{
    PHMS_INFO << "FaultTaskHandler::Init";
    thread_ = std::thread(&FaultTaskHandler::Run, this);
}

void
FaultTaskHandler::RegisterAnalysisCallback(std::function<void(Fault_t)> callback)
{
    PHMS_INFO << "FaultTaskHandler::RegisterAnalysisCallback";
    analysis_handler_ = callback;
}

void
FaultTaskHandler::RegisterRecorderCallback(std::function<void(Fault_t)> callback)
{
    PHMS_INFO << "FaultTaskHandler::RegisterRecorderCallback";
    record_handler_ = callback;
}

void
FaultTaskHandler::RegisterStrategyCallback(std::function<void(Fault_t)> callback)
{
    PHMS_INFO << "FaultTaskHandler::RegisterStrategyCallback";
    strategy_handler_ = callback;
}

void
FaultTaskHandler::RegisterDtcCallback(std::function<void(Fault_t)> callback)
{
    PHMS_INFO << "FaultTaskHandler::RegisterDtcCallback";
    dtc_handler_ = callback;
}

void
FaultTaskHandler::AddFault(const FaultTask task)
{
    PHMS_DEBUG << "FaultTaskHandler::AddFault id " << task.fault.faultId << " obj " << task.fault.faultObj;
    std::unique_lock<std::mutex> lck(mtx_);
    fault_queue_.push(task);
    cv_.notify_one();
}

void
FaultTaskHandler::Run()
{
    PHMS_INFO << "FaultTaskHandler::Run";
    PHMUtils::SetThreadName(fault_thread_name);
    while (!stop_flag_) {
        std::unique_lock<std::mutex> lck(mtx_);
        cv_.wait(lck);

        PHMS_DEBUG << "FaultTaskHandler::Run wake up!";

        if (stop_flag_) {
            PHMS_INFO << "FaultTaskHandler::Run end 1!";
            return;
        }

        while (!fault_queue_.empty()) {
            FaultTask task = fault_queue_.front();
            PHMS_INFO << "FaultTaskHandler::Run Fault domain: " << task.fault.faultDomain << " key: " << task.fault.faultId*100 + task.fault.faultObj << ", current queue size: " << fault_queue_.size();
            fault_queue_.pop();

            for (auto item : task.type_list) {
                if (item == kAnalysis && (analysis_handler_ != nullptr)) {
                    PHMS_DEBUG << "FaultTaskHandler::Run Analysis Branch";
                    analysis_handler_(task.fault);
                }

                if (item == kRecord && (record_handler_ != nullptr)) {
                    PHMS_DEBUG << "FaultTaskHandler::Run Record Branch";
                    record_handler_(task.fault);
                }

                if (item == kStrategy && (strategy_handler_ != nullptr)) {
                    PHMS_DEBUG << "FaultTaskHandler::Run Strategy Branch";
                    strategy_handler_(task.fault);
                }
            }
        }
    }

    PHMS_INFO << "FaultTaskHandler::Run end 2!";
}

void
FaultTaskHandler::DeInit()
{
    PHMS_INFO << "FaultTaskHandler::DeInit enter!";

    stop_flag_ = true;
    cv_.notify_one();

    if (thread_.joinable()) {
        thread_.join();
    }

    std::unique_lock<std::mutex> lck(mtx_);
    while (!fault_queue_.empty()) {
        fault_queue_.pop();
    }

    if (analysis_handler_ != nullptr) {
        analysis_handler_ = nullptr;
    }

    if (record_handler_ != nullptr) {
        record_handler_ = nullptr;
    }

    if (strategy_handler_ != nullptr) {
        strategy_handler_ = nullptr;
    }

    if (dtc_handler_ != nullptr) {
        dtc_handler_ = nullptr;
    }

    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
    PHMS_INFO << "FaultTaskHandler::DeInit done!";
}

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
