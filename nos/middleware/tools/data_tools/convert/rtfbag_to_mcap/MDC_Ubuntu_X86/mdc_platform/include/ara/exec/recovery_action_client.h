/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: The RecoveryActionClient provides the functionality for PlatformHealthManagement.
 *              This class is in draft state and it may be changed in future
 * Create: 2019-07-21
 */

#ifndef ARA_EXEC_RECOVERY_ACTION_CLIENT_H
#define ARA_EXEC_RECOVERY_ACTION_CLIENT_H

#include <cstdint>
#include <unistd.h>
#include <ara/core/future.h>
#include <ara/core/vector.h>
#include <ara/core/string.h>

namespace ara {
namespace exec {
enum class RecoveryActionReturnType : uint8_t {
    kSuccess      = 0U,
    kBusy         = 1U,
    kTimeout      = 2U,
    kGeneralError = 3U
};

struct Process {
    ara::core::String processName_;
    ara::core::String processState_;    // Process state which can be IDLE, STARTING, RUNNING, TERMINATING, TERMINATED
                                        // or ABORTED.
    ara::core::String functionGroup_;   // Function group to which the process belongs.
    ara::core::String functionGroupState_;   // Target function group state to which the process belongs.
    bool isFgChanging_ {true};
    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(processName_);
        fun(processState_);
        fun(functionGroup_);
        fun(functionGroupState_);
        fun(isFgChanging_);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(processName_);
        fun(processState_);
        fun(functionGroup_);
        fun(functionGroupState_);
        fun(isFgChanging_);
    }

    bool operator == (const ara::exec::Process& t) const
    {
        return ((processName_ ==  t.processName_) &&
                (processState_ ==  t.processState_) &&
                (functionGroup_ ==  t.functionGroup_) &&
                (functionGroupState_ ==  t.functionGroupState_) &&
                (isFgChanging_ ==  t.isFgChanging_));
    }
};

class RecoveryActionClient {
public:
    // ================================================================================================================
    // Function:    RecoveryActionClient
    // Description: Class default constructor
    // Params:      void
    // Return:      N/A
    // ================================================================================================================
    explicit RecoveryActionClient(std::string const user = "");

    // ================================================================================================================
    // Function:    ~RecoveryActionClient
    // Description: Class default destructor
    // Params:      void
    // Return:      N/A
    // ================================================================================================================
    ~RecoveryActionClient(void);

    // ================================================================================================================
    // Function:    GetProcessStates
    // Description: Get information about all processes
    // Params:      processes  - vector of all processes (out)
    // Return:      ara::core::Result<void>      - result
    // ================================================================================================================
    ara::core::Result<void> GetProcessStates(ara::core::vector<Process>& processes) const noexcept;

    // ================================================================================================================
    // Function:    ProcessRestart
    // Description: Request to restart a Process
    // Params:      std::string     process identifier(in)
    // Return:      ara::core::Future<void>  - result
    // ================================================================================================================
    using ProcessIdentifier = std::string;
    ara::core::Future<void> ProcessRestart(ProcessIdentifier const &processName,
        bool isForceRestart = false) const noexcept;

    // ================================================================================================================
    // Function:    GetProcessName
    // Description: Request to get process name according to its pid
    // Params:      pid_t                                 - pid
    // Return:      ara::core::Result<ara::core::String>  - result of process name
    // ================================================================================================================
    ara::core::Result<ara::core::String> GetProcessName(pid_t pid) const noexcept;

private:
    std::string user_ {};
};
} // namespace exec
} // namespace ara

#endif /* ARA_EXEC_RECOVERY_ACTION_CLIENT_H */

