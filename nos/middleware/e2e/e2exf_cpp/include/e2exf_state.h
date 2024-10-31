#ifndef E2EXF_STATE_H_
#define E2EXF_STATE_H_

#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>

#include "e2e/e2e/include/e2e_p04.h"
#include "e2e/e2e/include/e2e_p22.h"
namespace hozon {
namespace netaos {
namespace e2e {

using E2EXf_SMState = E2E_SMCheckStateType;

typedef struct {
    E2E_P04ProtectStateType P04ProtectState;
    E2E_P22ProtectStateType P22ProtectState;
} E2EXf_ProtectState;

typedef struct {
    E2E_P04CheckStateType P04CheckState;
    E2E_P22CheckStateType P22CheckState;
} E2EXf_CheckState;

class E2EXf_State final {
   public:
    E2EXf_State(const E2EXf_State&) = default;

    E2EXf_State& operator=(const E2EXf_State&) & = default;

    explicit E2EXf_State(const E2EXf_ProtectState& State);

    explicit E2EXf_State(const E2EXf_CheckState& State);

    E2EXf_State(const E2EXf_ProtectState& ProtectState, const E2EXf_CheckState& E2EXf_CheckState);

    E2EXf_ProtectState& GetProtectState() { return protect_state_; }

    E2EXf_CheckState& GetCheckState() { return check_state_; }

    E2EXf_SMState& GetSMState() { return sm_state_; }

    void SetProtectState(const E2EXf_ProtectState& ProtectState) { this->protect_state_ = ProtectState; }

    void SetCheckState(const E2EXf_CheckState& CheckState) { this->check_state_ = CheckState; }

    void SetSMStateType(const E2EXf_SMState& SMState) { this->sm_state_ = SMState; }
    E2EXf_State() = default;
    ~E2EXf_State() = default;

   private:
    E2EXf_ProtectState protect_state_{{0}, {0}};

    // std::shared_ptr<E2EXf_CheckState> CheckState = std::make_shared<E2EXf_CheckState>(E2EXf_CheckState{{E2E_P04STATUS_NONEWDATA, 0}, {E2E_P22STATUS_NONEWDATA, 0}});
    E2EXf_CheckState check_state_{E2EXf_CheckState{{E2E_P04STATUS_NONEWDATA, 0}, {E2E_P22STATUS_NONEWDATA, 0}}};

    // std::shared_ptr<E2EXf_SMState> SMState = std::make_shared<E2EXf_SMState>(E2E_SMCheckStateType{new uint8_t, 0, 0, 0, E2E_SM_DEINIT});
    E2EXf_SMState sm_state_{E2E_SMCheckStateType{new uint8_t, 0, 0, 0, E2E_SM_DEINIT}};

    static E2EXf_State* instancePtr_;
    static std::mutex mtx_;
};

}  // namespace e2e
}  // namespace netaos
}  // namespace hozon

#endif