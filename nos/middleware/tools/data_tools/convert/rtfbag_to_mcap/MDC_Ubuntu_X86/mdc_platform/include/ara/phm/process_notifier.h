/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description:  definition
 * Create: ProcessNotifier 2020-12-10
 */
#ifndef ARA_PHM_PROCESS_NOTIFIER_H
#define ARA_PHM_PROCESS_NOTIFIER_H

#include <string>
#include <rtf/com/rtf_com.h>

namespace ara {
namespace phm {
using CommunicationMode = std::set<rtf::com::TransportMode>;
/**
 * @defgroup ProcessNotifier ProcessNotifier
 * @brief Container for all ProcessNotifier objects.
 * @ingroup ProcessNotifier
 */
class State final {
public:
    /**
     * @ingroup ProcessNotifier
     * @brief Constructor of StateState.
     * @param[in] processState The state of the process to report.
     * @param[in] functionGroupState The function group state of the process to report.
     * @return State
     * @req{AR-iAOS-RTF-RTFPHM-00019,
     * The RTFPHM supports the interface for notifying process status changes.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00019
     * }
     */
    State(ara::core::String const &processState, ara::core::String const &functionGroupState)
        : processState_(processState), functionGroupState_(functionGroupState)
    {}
    /**
     * @ingroup ProcessNotifier
     * @brief Destructor of State.
     * @req{AR-iAOS-RTF-RTFPHM-00019,
     * The RTFPHM supports the interface for notifying process status changes.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00019
     * }
     */
    ~State() = default;
    /**
     * @ingroup ProcessNotifier
     * @brief Copy constructor function of State.
     * @param[in] Const ref of State.
     * @req{AR-iAOS-RTF-RTFPHM-00019,
     * The RTFPHM supports the interface for notifying process status changes.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00019
     * }
     */
    State(State const&) = default;
    /**
     * @ingroup ProcessNotifier
     * @brief Move constructor function of State.
     * @param[in] Ref of State.
     * @req{AR-iAOS-RTF-RTFPHM-00019,
     * The RTFPHM supports the interface for notifying process status changes.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00019
     * }
     */
    State(State&&) = default;
    /**
     * @ingroup ProcessNotifier
     * @brief Assignment constructor function of State.
     * @param[in] Const ref of State.
     * @req{AR-iAOS-RTF-RTFPHM-00019,
     * The RTFPHM supports the interface for notifying process status changes.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00019
     * }
     */
    State& operator=(State const&) & = default;
    /**
     * @ingroup ProcessNotifier
     * @brief Move assignment constructor function of State.
     * @param[in] Ref of State.
     * @req{AR-iAOS-RTF-RTFPHM-00019,
     * The RTFPHM supports the interface for notifying process status changes.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00019
     * }
     */
    State& operator=(State&&) & = default;
    /**
     * @ingroup ProcessNotifier
     * @brief Get the process state.
     * @return core::String
     * @req{AR-iAOS-RTF-RTFPHM-00019,
     * The RTFPHM supports the interface for notifying process status changes.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00019
     * }
     */
    core::String const &GetProcessState() const
    {
        return processState_;
    }
    /**
     * @ingroup ProcessNotifier
     * @brief Get the function group state.
     * @return core::String
     * @req{AR-iAOS-RTF-RTFPHM-00019,
     * The RTFPHM supports the interface for notifying process status changes.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00019
     * }
     */
    core::String const &GetFunctionGroupState() const
    {
        return functionGroupState_;
    }
private:
    ara::core::String processState_;
    ara::core::String functionGroupState_;
};
/* AXIVION Next Line AutosarC++19_03-A12.0.1, AutosarC++19_03-A0.1.6 : Standard external interface,
 * can't add other function, the unused type is offered to the user [] */
class ProcessNotifier final {
public:
    /**
     * @ingroup ProcessNotifier
     * @brief Constructor of ProcessNotifier.
     * @return ProcessNotifier
     * @req{AR-iAOS-RTF-RTFPHM-00019,
     * The RTFPHM supports the interface for notifying process status changes.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00019
     * }
     */
    ProcessNotifier(CommunicationMode const &mode = {rtf::com::TransportMode::UDP});
    /**
     * @ingroup ProcessNotifier
     * @brief Destructor of ProcessNotifier.
     * @req{AR-iAOS-RTF-RTFPHM-00019,
     * The RTFPHM supports the interface for notifying process status changes.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00019
     * }
     */
    ~ ProcessNotifier() = default;
    /**
     * @ingroup ProcessNotifier
     * @brief Report the process state, sequence number to phm server.
     * @par Description
     * The process state change notification can be used by the Platform Health Manager
     * to detemine which Supervision Entity is activated or deactivated
     * @param[in] processName The name of the process.
     * @param[in] state The process state.
     * @param[in] seqNum The sequence number.
     * @return void
     * @req{AR-iAOS-RTF-RTFPHM-00019,
     * The RTFPHM supports the interface for notifying process status changes.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00019
     * }
     */
    void ProcessChanged(std::string const &processName, State const &state, uint64_t seqNum) const;

private:
    rtf::com::InstanceId instanceId_ {};
};
} // namespace phm
} // namespace ara
#endif

