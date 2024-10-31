/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: emchecker api
 * Create: 2021-04-28
 */
#ifndef VRTF_CHECKER_STATE_MANAGER_H
#define VRTF_CHECKER_STATE_MANAGER_H

#include "ara/exec/recovery_action_client.h"

namespace ara     {
namespace phmplugin {
/**
 * @defgroup Plugin Plugin
 * @brief Container for all CheckerInterface objects.
 * @ingroup Plugin
 */
/* AXIVION Next Line AutosarC++19_03-A0.1.6 : The class is external interface, it is offered to the user. */
class CheckerInterface final {
public:
    /**
     * @ingroup Plugin
     * @brief Constructor of CheckerInterface.
     * @return CheckerInterface
     * @req{SAR-iAOS3-RTF-RTFPHM-00000,
     * RTFPHM supports integration and triggers EMChecker detection.,
     * D,
     * SDR-iAOS3-RTF-RTFPHM-00000
     * }
     */
    CheckerInterface();
    /**
     * @ingroup Plugin
     * @brief Destructor of CheckerInterface.
     * @req{SAR-iAOS3-RTF-RTFPHM-00000,
     * RTFPHM supports integration and triggers EMChecker detection.,
     * D,
     * SDR-iAOS3-RTF-RTFPHM-00000
     * }
     */
    ~CheckerInterface(void) = default;
    /**
     * @ingroup Plugin
     * @brief Initialize the checker.
     * @param[in] rootPath The root path of em cofig file.
     * @param[in] configPath The machine config file path.
     * @param[in] appBinCfgPath The config file paths of applications.
     * @param[in] crcIntensity CRC validation level.
     * @return bool
     * @retval true The checker is initialized successfully.
     * @retval false Failed to initialize the checker.
     * @req{SAR-iAOS3-RTF-RTFPHM-00000,
     * RTFPHM supports integration and triggers EMChecker detection.,
     * D,
     * SDR-iAOS3-RTF-RTFPHM-00000
     * }
     */
    /* AXIVION disable style AutosarC++19_03-A5.1.1 : weak crc option for default para value. */
    bool Init(std::string const &rootPath, std::string const &configPath,
              const std::vector<std::string>& appBinCfgPath, std::string const &crcIntensity = "weak") const;
    /* AXIVION enable style AutosarC++19_03-A5.1.1 */
    /**
     * @ingroup Plugin
     * @brief Set the process states.
     * @param[in] processes The process states to be set.
     * @return bool
     * @retval true The process states is set successfully.
     * @retval false Failed to set the process states.
     * @req{SAR-iAOS3-RTF-RTFPHM-00000,
     * RTFPHM supports integration and triggers EMChecker detection.,
     * D,
     * SDR-iAOS3-RTF-RTFPHM-00000
     * }
     */
    bool SetProcessStates(ara::core::vector<ara::exec::Process> const &processes);
    /**
     * @ingroup Plugin
     * @brief Run the checker to check process states.
     * @param[in] processName The process name of the process to check.
     * @param[in] processState The process state of the process to check.
     * @param[in] functionGroupStates The function group state of the process to check.
     * @return bool
     * @retval true The process states is normal.
     * @retval false The process states is abnormal.
     * @req{SAR-iAOS3-RTF-RTFPHM-00000,
     * RTFPHM supports integration and triggers EMChecker detection.,
     * D,
     * SDR-iAOS3-RTF-RTFPHM-00000
     * }
     */
    bool Run(std::string const &processName,  std::string const &processState, std::string &functionGroupStates) const;
    /**
     * @ingroup Plugin
     * @brief Deinitialize the checker.
     * @return void
     * @req{SAR-iAOS3-RTF-RTFPHM-00000,
     * RTFPHM supports integration and triggers EMChecker detection.,
     * D,
     * SDR-iAOS3-RTF-RTFPHM-00000
     * }
     */
    void DeInit();
    CheckerInterface(CheckerInterface const&) = default ;
    CheckerInterface(CheckerInterface&&) = default ;
    CheckerInterface& operator=(CheckerInterface const&) & = default ;
    CheckerInterface& operator=(CheckerInterface&&) & = default ;

private:
    bool ifInit_ {false};
};
}
}
#endif // VRTF_CHECKER_STATE_MANAGER_H
