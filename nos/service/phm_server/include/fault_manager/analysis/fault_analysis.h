#ifndef FAULT_ANALYSIS_H
#define FAULT_ANALYSIS_H

#include <mutex>
#include "phm_server/include/common/phm_server_def.h"
#include "phm_server/include/common/time_manager.h"

namespace hozon {
namespace netaos {
namespace phm_server {

/********************************************************************************/
class FaultAnalysis final {
public:
    static FaultAnalysis *getInstance();

    void Init();
    void DeInit();

    void AnalysisFaultCallback(Fault_t fault);
    void NotifyAnalysisFaultData(const Fault_t& faultData);
    void UpdateAnalyFile();

    void UpdateAnalyList(const Fault_t& recvFaultData, AnalysisFault& savedFaultData);
    void SaveFaultToAnalyList(const Fault_t& faultData);
    void UpdateAnalyStatusList(const Fault_t& faultData, AnalysisFaultStatus& analysisFaultStatus);
    void SaveAnalyStatusList(const Fault_t& faultData);
    void UpdateAnalyNonstandardList(const std::string faultDomain, AnalysisFaultStatus& analysisFaultStatus);

    std::string ReadAnalyCountData();
    std::string ReadAnalysNonstandardData();
    std::string ReadAnalyOverCountData();
    std::string ReadStartupFaultData();
    std::string ReadPresentFaultData();
    std::string GetFaultAnalyData();

private:
    //! 构造函数
    FaultAnalysis();
    //! 析构函数
    ~FaultAnalysis();

    void InitTimeoutCallback(void* data);

    static FaultAnalysis *instancePtr_;
    static std::mutex mtx_;
    static std::mutex analy_list_mtx_;
    static std::mutex analy_status_list_mtx_;
    static std::mutex analy_nonstandard_list_mtx_;
    static std::mutex system_check_list_mtx_;
    static std::mutex update_analysisfile_mtx_;

    std::shared_ptr<TimerManager> time_mgr_;
    int init_timer_fd_;
    std::string debug_switch_;

    std::vector<AnalysisFault> analy_list_;
    std::vector<AnalysisFaultStatus> analy_status_list_;
    std::vector<AnalisysNonstandard> analy_nonstandard_list_;
    std::vector<AnalisysOverCount> analy_over_count_list_;
    std::vector<SystemCheckFaultInfo> system_check_list_;
    std::vector<uint32_t> current_fault_list_;
};

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
#endif  // FAULT_ANALYSIS_H
