/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: StateManager Header
 * Create: 2020-6-6
 */
#ifndef ARA_PHM_PHM_STATE_MANAGER_H
#define ARA_PHM_PHM_STATE_MANAGER_H

#include <string>
#include <functional>
#include <map>
#include <vector>
#include <atomic>
#include <rtf/com/rtf_com.h>

namespace ara {
namespace phm {
enum class CrcCheck : uint8_t {
    NO = 0U, // ignore crc code
    WEAK = 1U, // check if file has crc code
    STRONG = 2U, // check failed if file don't have crc code
};

using CommunicateMode = std::set<rtf::com::TransportMode>;
/**
 * @defgroup  DetectorSkeleton  DetectorSkeleton
 * @brief Container for all healthy detector objects.
 * @ingroup  DetectorSkeleton  DetectorSkeleton
 */
class PhmStateManager final {
public:
    /**
     * @ingroup DetectorSkeleton
     * @brief Destructor of PhmStateManager.
     *
     * @req{AR-iAOS-RTF-RTFPHM-00057,
     * The RTFPHM supports enabling and disabling the monitoring service.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00057
     * }
     */
    ~PhmStateManager();
    PhmStateManager(const PhmStateManager &) = delete;
    PhmStateManager(PhmStateManager &&) = delete;
    PhmStateManager &operator=(const PhmStateManager &) & = delete;
    PhmStateManager &operator=(PhmStateManager &&) & = delete;
    /**
     * @ingroup DetectorSkeleton
     * @brief Get instance of PhmStateManager.
     * @return PhmStateManager
     *
     * @req{AR-iAOS-RTF-RTFPHM-00057,
     * The RTFPHM supports enabling and disabling the monitoring service.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00057
     * }
     */
    static PhmStateManager &GetInstance();
    /**
     * @ingroup DetectorSkeleton
     * @brief Set thread num.
     * @param[in] number the num of thread.
     * @return bool
     * @retval true set success.
     * @retval false set failed.
     *
     * @req{AR-iAOS-RTF-RTFPHM-00057,
     * The RTFPHM supports enabling and disabling the monitoring service.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00057
     * }
     */
    bool SetConcurrentNumber(uint32_t number, uint32_t actionThreadNum = 1U);
    /**
     * @ingroup DetectorSkeleton
     * @brief Get thread num.
     * @return uint32_t
     *
     * @req{AR-iAOS-RTF-RTFPHM-00057,
     * The RTFPHM supports enabling and disabling the monitoring service.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00057
     * }
     */
    inline uint32_t GetConcurrentNumber() const
    {
        return concurrentNumber_;
    }
    /**
     * @ingroup DetectorSkeleton
     * @brief Init phm.[SWS_HM_00455]
     * @par Description
     * 1. This interface provide a method to initialize the service,
     * 2. User can set shm file user when using shm transport,
     * 3. User can set transport mode, plugin name and crcOption.
     *
     * @param[in] user the user of shm file.
     * @param[in] mode the communication mode.
     * @param[in] pluginName the name of plugin.
     * @param[in] crcOption the option of crc.
     * @return bool
     * @retval true init success.
     * @retval false init failed.
     *
     * @req{AR-iAOS-RTF-RTFPHM-00057,
     * The RTFPHM supports enabling and disabling the monitoring service.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00057
     * }
     */
    bool Init(std::string const &user = "",
              CommunicateMode const &mode = {rtf::com::TransportMode::UDP},
              std::vector<std::string> const &pluginName = {}, CrcCheck const crcOption = CrcCheck::WEAK);
    /**
     * @ingroup DetectorSkeleton
     * @brief DeInit phm.[SWS_HM_00456]
     * @par Description
     * It provide a method to deinitialize the service and release resource.
     *
     * @return void
     *
     * @req{AR-iAOS-RTF-RTFPHM-00057,
     * The RTFPHM supports enabling and disabling the monitoring service.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00057
     * }
     */
    void DeInit();
    /**
     * @ingroup Communication
     * @brief Enable e2e of se.
     * @param[in] supervisedEntityDataId the data id of se.
     * @return void
     *
     * @req{SAR-iAOS3-RTF-RTFPHM-00005,
     * RTFPHM supports the function security of the checkpoint interface.,
     * D,
     * SDR-iAOS3-RTF-RTFPHM-00005
     * }
     */
    static void EnableSEE2E(std::uint32_t supervisedEntityDataId);
    /**
     * @ingroup Communication
     * @brief Enable e2e of hc.
     * @param[in] healthChannelDataId the data id of hc.
     * @return void
     *
     * @req{SAR-iAOS3-RTF-RTFPHM-00006,
     * RTFPHM supports the function security of the HealthChannel interface.,
     * D,
     * SDR-iAOS3-RTF-RTFPHM-00006
     * }
     */
    static void EnableHCE2E(std::uint32_t healthChannelDataId);
    /**
     * @ingroup DetectorSkeleton
     * @brief Query the result of rule.
     * @par Description
     * 1. Get the logical expression of phm rule;
     * 2. Return all result of argument in logical expression.
     * @param[in] ruleName the name of rule.
     * @return std::map<std::string, bool>
     *
     * @req{AR-iAOS-RTF-RTFPHM-00032,
     * The RTFPHM supports the fault query interface.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00032
     * }
     */
    static std::map<std::string, bool> QueryRuleResults(std::string const &ruleName);
    /**
     * @ingroup DetectorSkeleton
     * @brief Register the call back of process states.
     * @param[in] cb the call back of process states.
     * @return void
     *
     * @req{AR-iAOS-RTF-RTFPHM-00054,
     * The RTFPHM provides the interface for registering the callback function for obtaining the process status list.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00054
     * }
     */
    void RegisterProcessStatesCallback(const std::function<std::vector<std::pair<std::string,
        std::string>>(void)> &cb) const;
    /**
     * @ingroup DetectorSkeleton
     * @brief Set the state of process.
     * @param[in] processName the name of process.
     * @param[in] processState the state of process.
     * @return void
     *
     * @req{AR-iAOS-RTF-RTFPHM-00054,
     * The RTFPHM provides the interface for registering the callback function for obtaining the process status list.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00054
     * }
     */
    static void SetProcessState(std::string const &processName, std::string const &processState);
    /**
     * @ingroup DetectorSkeleton
     * @brief Set the path of em checker.
     * @param[in] rootPath the path of root.
     * @param[in] machinePath the path of machine.
     * @param[in] appBinCfgPath the path of app.
     * @return void
     *
     * @req{SAR-iAOS3-RTF-RTFPHM-00000,
     * RTFPHM supports integration and triggers EMChecker detection.,
     * D,
     * SDR-iAOS3-RTF-RTFPHM-00000
     * }
     */
    void SetEmCheckPath(std::string const &rootPath, std::string const &machinePath,
        const std::vector<std::string> &appBinCfgPath);
    /**
     * @ingroup DetectorSkeleton
     * @brief Register the call back of recovery.
     * @param[in] cb the call back of recovery.
     * @return void
     *
     * @req{AR-iAOS-RTF-RTFPHM-00055,
     * The RTFPHM supports ADS application recovery.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00055
     * }
     */
    void RegisterRecoveryCallback(const std::function<void(std::string ruleInfo, std::string actionName,
        std::string portName, std::string methodName, std::string processName)> &cb) const;
    /**
     * @ingroup DetectorSkeleton
     * @brief Register the call back of safety.
     * @param[in] cb the call back of safety.
     * @return void
     *
     * @req{SAR-iAOS3-RTF-RTFPHM-00000,
     * RTFPHM supports integration and triggers EMChecker detection.,
     * D,
     * SDR-iAOS3-RTF-RTFPHM-00000
     * }
     */
    void RegisterSafetyCallback(const std::function<void(std::string, std::string)> &cb) const;

private:
    /**
     * @ingroup DetectorSkeleton
     * @brief Constructor of PhmStateManager.
     *
     * @req{AR-iAOS-RTF-RTFPHM-00057,
     * The RTFPHM supports enabling and disabling the monitoring service.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00057
     * }
     */
    PhmStateManager();
    /**
     * @ingroup DetectorSkeleton
     * @brief Release resource.
     * @par Description
     * 1. Stop offer service and thread pool;
     * 2. DeInit global and em checker plugin;
     * 3. Clear restored data.
     * @return void
     *
     * @req{AR-iAOS-RTF-RTFPHM-00057,
     * The RTFPHM supports enabling and disabling the monitoring service.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00057
     * }
     */
    static void ReleaseResource();
    /**
     * @ingroup DetectorSkeleton
     * @brief Get process list from em, will update processList shared by multiple threads.
     * @return bool
     * @retval true get success.
     * @retval false get failed.
     *
     * @req{AR-iAOS-RTF-RTFPHM-00021,
     * The RTFPHM supports obtaining process status information.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00021
     * }
     *
     * @startuml
     * start
     * :Before get process list, set hasGet to false, checkResult=true;
     * :Call the GetProcessStates interface of EM;
     * if (get success?) then (yes)
     *   if (emcheck is not nullptr?) then (yes)
     *     :Call SetProcessStates of emchecker;
     *     if (check success?) then (yes)
     *       :Update process list;
     *       :Set hasGet to true;
     *     else (no)
     *       :Set checkResult to false;
     *     endif
     *     else (no)
     *       :Update process list;
     *   endif
     * else (no)
     *   :Get process list failed;
     * endif
     * :Return checkResult;
     * stop
     * @enduml
     *
     */
    static bool GetProcessList();
    /**
     * @ingroup DetectorSkeleton
     * @brief Get process list from call back.
     * @return void
     *
     * @req{AR-iAOS-RTF-RTFPHM-00021,
     * The RTFPHM supports obtaining process status information.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00021
     * }
     *
     */
    static void GetProcessListFromCb() ;
    /**
     * @ingroup DetectorSkeleton
     * @brief Clear loaded data.
     * @par Description
     * 1. Clear loaded data from json file;
     * 2. Clear call back function.
     * @return void
     *
     * @req{AR-iAOS-RTF-RTFPHM-00052,
     * The RTFPHM supports dynamic modification of the JSON configuration file.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00052
     * }
     */
    static void ClearData();
    /**
     * @ingroup Communication
     * @brief Is the communicate mode valid.
     * @par Description
     * Only SHM + UDP or SHM + UDP + ICC is valid.
     * @return bool
     * @retval true is valid.
     * @retval false is not valid.
     *
     * @req{SAR-iAOS-RTF-RTFPHM-00012,
     * RTFPHM server supports UDP, SHM, and ICC communication modes.,
     * D,
     * SDR-iAOS-RTF-RTFPHM-00012
     * }
     */
    static bool IsCommunicateModeValid(CommunicateMode const &mode);
    /**
     * @ingroup DetectorSkeleton
     * @brief Init plugin.
     * @par Description
     * 1. Check crcOption;
     * 2. Init em plugin.
     * @param[in] pluginName the name of plugin.
     * @param[in] crcOption the option of crc.
     * @return bool
     * @retval true init success.
     * @retval false init failed.
     *
     * @req{SAR-iAOS3-RTF-RTFPHM-00000,
     * RTFPHM supports integration and triggers EMChecker detection.,
     * D,
     * SDR-iAOS3-RTF-RTFPHM-00000
     * }
     */
    bool InitPlugin(const std::vector<std::string> &pluginName, CrcCheck const crcOption) const;
    /**
     * @ingroup DetectorSkeleton
     * @brief Init em checker.
     * @param[in] crcOption the option of crc.
     * @return bool
     * @retval true init success.
     * @retval false init failed.
     *
     * @req{SAR-iAOS3-RTF-RTFPHM-00000,
     * RTFPHM supports integration and triggers EMChecker detection.,
     * D,
     * SDR-iAOS3-RTF-RTFPHM-00000
     * }
     */
    bool InitEmChecker(std::string const &crcOption) const;
    /**
     * @ingroup DetectorSkeleton
     * @brief Load and check json file.
     * @return bool
     * @retval true check success.
     * @retval false check failed.
     *
     * @req{AR-iAOS-RTF-RTFPHM-00052,
     * The RTFPHM supports dynamic modification of the JSON configuration file.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00052
     * }
     */
    static bool JsonCheck(CrcCheck const crcOption);

    std::string rootPath_  {};
    std::string machinePath_ {};
    std::vector<std::string> appBinCfgPath_ {};
    std::atomic_bool isInitialled_ {false};
    uint32_t concurrentNumber_ {1U};
    uint32_t actionThreadNumber_ {1U};
    std::mutex mutex_ {};
    bool hasSetEmCheckPath_ {false};
};
}
}
#endif

