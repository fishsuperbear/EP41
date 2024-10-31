
#include "yaml-cpp/yaml.h"
#include "common.h"
#include "phm/include/phm_client.h"
#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>


class PhmFuncTest
{
public:
    PhmFuncTest();
    ~PhmFuncTest();

    void Init(const std::string& phmConfigPath);
    void DeInit();
    void Start();
    void Stop();
    bool Up(int);
    void Down(int);

    void LoadInitYamlConfig();
    void LoadData(const YAML::Node& node, const std::string& type);
    void ParseRuleData(const YAML::Node& node, const std::string& type);
    bool checkIsEqualCurFault();
    bool checkIsRecFault();

    void checkServiceStatus();
    static void PhmServiceStatusCallbak(const bool status);
    static void FmFaultRecCallbak(hozon::netaos::phm::ReceiveFault data);
    static void FmFaultRecCallbak2(hozon::netaos::phm::ReceiveFault data);
    static void FmFaultRecCallbak3(hozon::netaos::phm::ReceiveFault data);
    void ReportFault(hozon::netaos::phm::SendFault_t& sendFault);
    void ReportFault(uint32_t faultId, uint8_t faultObj, uint8_t faultStatus);
    void ReportFaultWithTimeDebounce(uint32_t faultId, uint8_t faultObj, uint8_t faultStatus);
    void ReportFaultWithCountDebounce(uint32_t faultId, uint8_t faultObj, uint8_t faultStatus);
    void ReportFaultAndOccurDtc() {}
    char* GetJsonAll(const char *fname);
    int32_t ParseFaultJson();
    void MultiThreadReportTest();

    // main if
    bool AutoTest();

    // case 1
    void HmAliveNormalAutoTest();
    void HmDeadlineNormalAutoTest();
    void HmLogicNormalAutoTest();

    // case 2/3
    void HmAliveAbnormalAutoTest();
    void HmDeadlineAbnormalAutoTest();
    void HmLogicAbnormalAutoTest();

    // case 4 TODO
    void HmMonitorProcessTest();

    // case 8
    void FaultListenerTest();

    // case 9
    void DebounceTestNoFault();

    // case 10/11
    void DebounceTestGenerateFault();

    // case 12->16 TODO

    // case 23 TODO
    void ResFileGenerateTest() {}

    // 26
    void FmDataCollectAutoTest();

    // case 27 TODO
    void FmDtcGenerateTest() {}

    // case 28 TODO
    void FmLinterDiagClearCmdTest() {}

    void ReportAllFaultAutoTest();

    void printResult();

    void FuctionRandomCallTest();

    static std::atomic<bool> m_bEqualCurrentFault;
    static long long iCount;
    long long all_success_count = 0;
    long long all_fail_count = 0;
    long long comm_fail_count = 0;
    std::shared_ptr<hozon::netaos::phm::PHMClient> m_spPhmClient;
};
