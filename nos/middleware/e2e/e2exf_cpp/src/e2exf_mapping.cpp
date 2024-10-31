#include "e2e/e2exf_cpp/include/e2exf_mapping.h"

#include <utility>
namespace hozon {
namespace netaos {
namespace e2e {

E2EXf_Mapping* E2EXf_Mapping::instanceptr_ = nullptr;
std::mutex E2EXf_Mapping::mtx_;

E2EXf_Mapping* E2EXf_Mapping::Instance() {
    if (nullptr == instanceptr_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instanceptr_) {
            instanceptr_ = new E2EXf_Mapping();
        }
    }
    return instanceptr_;
}

void E2EXf_Mapping::bind(const E2EXf_Index& index, const E2EXf_Config& config) {
    // E2EXf_State state;
    indexmap_[index] = std::make_pair(config, *(new E2EXf_State));
}

E2EXf_ProtectState& E2EXf_Mapping::GetProtectState(const E2EXf_Index& index) {
    return indexmap_[index].second.GetProtectState();  //
}

E2EXf_CheckState& E2EXf_Mapping::GetCheckState(const E2EXf_Index& index) {
    return indexmap_[index].second.GetCheckState();  //
}

E2EXf_SMState& E2EXf_Mapping::GetSMState(const E2EXf_Index& index) {
    return indexmap_[index].second.GetSMState();  //
}

const E2EXf_ConfigType& E2EXf_Mapping::GetE2EConfig(const E2EXf_Index& index) {
    return indexmap_[index].first.GetE2EXfConfig();  //
}

const E2EXf_SMConfigType& E2EXf_Mapping::GetE2ESMConfig(const E2EXf_Index& index) {
    return indexmap_[index].first.GetSMConfig();  //
}

E2E_PCheckStatusType E2EXf_Mapping::GetProfileCheckStatus(const E2EXf_Index& index) {
    auto state = indexmap_[index].second.GetCheckState();
    switch (indexmap_[index].first.GetE2EXfConfig().Profile)
    {
    case E2EXf_Profile::PROFILE04:
        return E2E_P04MapStatusToSM(Std_ReturnType::E2E_E_OK,state.P04CheckState.Status);
        break;

    case E2EXf_Profile::PROFILE22:
        return E2E_P22MapStatusToSM(Std_ReturnType::E2E_E_OK,state.P22CheckState.Status);
        break;

    case E2EXf_Profile::PROFILE22_CUSTOM:
        return E2E_P22MapStatusToSM(Std_ReturnType::E2E_E_OK,state.P22CheckState.Status);
        break;  

    default:
        break;
    }

    return E2E_PCheckStatusType::E2E_P_RESERVED;
}
}  // namespace e2e
}  // namespace netaos
}  // namespace hozon