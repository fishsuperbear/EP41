#ifndef PHM_TASK_MANAGER_H
#define PHM_TASK_MANAGER_H

#include <mutex>
#include <queue>
#include <thread>
#include <functional>
#include <condition_variable>
#include "phm_server/include/common/phm_server_def.h"

namespace hozon {
namespace netaos {
namespace phm_server {

enum FaultTaskType {
    kAnalysis = 0x00,
    kRecord = 0x01,
    kStrategy = 0x02,
    kDtcMapping = 0x03
};

struct FaultTask {
    std::vector<FaultTaskType> type_list;
    Fault_t fault;
};

class FaultTaskHandler {

public:
    static FaultTaskHandler* getInstance();

    void Init();
    void DeInit();

    void RegisterAnalysisCallback(std::function<void(Fault_t)> callback);
    void RegisterRecorderCallback(std::function<void(Fault_t)> callback);
    void RegisterStrategyCallback(std::function<void(Fault_t)> callback);
    void RegisterDtcCallback(std::function<void(Fault_t)> callback);
    void AddFault(const FaultTask task);
    void Run();

private:
    FaultTaskHandler();
    FaultTaskHandler(const FaultTaskHandler &);
    FaultTaskHandler & operator = (const FaultTaskHandler &);

private:
    static std::mutex mtx_;
    static FaultTaskHandler* instance_;

    bool stop_flag_;
    std::condition_variable cv_;
    std::thread thread_;
    std::vector<std::thread> thread_pool_;
    std::queue<FaultTask> fault_queue_;

    std::function<void(Fault_t)> analysis_handler_;
    std::function<void(Fault_t)> record_handler_;
    std::function<void(Fault_t)> strategy_handler_;
    std::function<void(Fault_t)> dtc_handler_;
};

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_TASK_MANAGER_H