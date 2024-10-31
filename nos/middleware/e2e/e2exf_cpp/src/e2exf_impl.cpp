#include "e2e/e2exf_cpp/include/e2exf_impl.h"

#include "e2e/e2exf_cpp/src/e2exf_customfunc.h"

namespace hozon {
namespace netaos {
namespace e2e {

bool AddE2EXfConfig(const E2EXf_Index& Index, const E2EXf_Config& Config) {
    hozon::netaos::e2e::E2EXf_Mapping::Instance()->bind(Index, Config);
    switch (Config.GetE2EXfConfig().Profile) {
        case E2EXf_Profile::PROFILE04:
            E2EXf_P04_ProtectInit(&hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetProtectState(Index).P04ProtectState);
            E2EXf_P04_CheckInit(&hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetCheckState(Index).P04CheckState);
            break;

        case E2EXf_Profile::PROFILE22:
            E2EXf_P22_ProtectInit(&hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetProtectState(Index).P22ProtectState);
            E2EXf_P22_CheckInit(&hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetCheckState(Index).P22CheckState);
            break;

        case E2EXf_Profile::PROFILE22_CUSTOM:
            E2EXf_P22_ProtectInit(&hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetProtectState(Index).P22ProtectState);
            E2EXf_P22_CheckInit(&hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetCheckState(Index).P22CheckState);
            break;

        default:
            return false;
            break;
    }
    return true;
}

ProtectResult E2EXf_Protect(const E2EXf_Index& Index, Payload& Buffer, const std::uint32_t InputBufferLength) {
    uint32_t BufferLength = 0;
    uint8_t protect_ret = E_SAFETY_HARD_RUNTIMEERROR;
    switch (hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index).Profile) {
        case E2EXf_Profile::PROFILE04:
            protect_ret = E2EXf_P04(&Buffer[0], &BufferLength, nullptr, InputBufferLength, &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index),
                                    &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetProtectState(Index).P04ProtectState, Std_TransformerForwardCode::E_OK);
            break;

        case E2EXf_Profile::PROFILE22:
            protect_ret = E2EXf_P22(&Buffer[0], &BufferLength, nullptr, InputBufferLength, &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index),
                                    &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetProtectState(Index).P22ProtectState, Std_TransformerForwardCode::E_OK);
            break;

        case E2EXf_Profile::PROFILE22_CUSTOM:
            protect_ret = E2EXf_P22_CUSTOM(&Buffer[0], &BufferLength, nullptr, InputBufferLength, &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index),
                                           &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetProtectState(Index).P22ProtectState, Std_TransformerForwardCode::E_OK, Profile22_custom_before,
                                           Profile22_custom_after);
            break;

        default:
            return ProtectResult::HardRuntimeError;
            break;
    }
    if (protect_ret == E_SAFETY_HARD_RUNTIMEERROR) return ProtectResult::HardRuntimeError;
    return ProtectResult::E_OK;
}

ProtectResult E2EXf_Protect_OutOfPlace(const E2EXf_Index& Index, const Payload& InputBuffer, Payload& Buffer, const std::uint32_t InputBufferLength, std::uint32_t& BufferLength) {
    uint8_t protect_ret = E_SAFETY_HARD_RUNTIMEERROR;
    uint8_t* buffer = new uint8_t[InputBufferLength + hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index).headerLength];
    switch (hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index).Profile) {
        case E2EXf_Profile::PROFILE04:
            protect_ret = E2EXf_P04(buffer, &BufferLength, &InputBuffer[0], InputBufferLength, &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index),
                                    &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetProtectState(Index).P04ProtectState, Std_TransformerForwardCode::E_OK);
            break;

        case E2EXf_Profile::PROFILE22:
            protect_ret = E2EXf_P22(buffer, &BufferLength, &InputBuffer[0], InputBufferLength, &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index),
                                    &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetProtectState(Index).P22ProtectState, Std_TransformerForwardCode::E_OK);
            break;

        case E2EXf_Profile::PROFILE22_CUSTOM:
            protect_ret = E2EXf_P22_CUSTOM(buffer, &BufferLength, &InputBuffer[0], InputBufferLength, &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index),
                                           &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetProtectState(Index).P22ProtectState, Std_TransformerForwardCode::E_OK, Profile22_custom_before,
                                           Profile22_custom_after);
            break;

        default:
            return ProtectResult::HardRuntimeError;
            break;
    }
    Buffer.clear();
    Buffer.insert(Buffer.begin(), buffer, buffer + BufferLength);
    delete[] buffer;
    if (protect_ret == E_SAFETY_HARD_RUNTIMEERROR) return ProtectResult::HardRuntimeError;
    return ProtectResult::E_OK;
}

ProtectResult E2EXf_Forward(const E2EXf_Index& Index, Payload& Buffer, const std::uint32_t InputBufferLength, Std_TransformerForwardCode ForwardCode) {
    uint32_t BufferLength = 0;
    uint8_t protect_ret = E_SAFETY_HARD_RUNTIMEERROR;
    switch (hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index).Profile) {
        case E2EXf_Profile::PROFILE04:
            protect_ret = E2EXf_P04(&Buffer[0], &BufferLength, nullptr, InputBufferLength, &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index),
                                    &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetProtectState(Index).P04ProtectState, ForwardCode);
            break;

        case E2EXf_Profile::PROFILE22:
            protect_ret = E2EXf_P22(&Buffer[0], &BufferLength, nullptr, InputBufferLength, &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index),
                                    &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetProtectState(Index).P22ProtectState, ForwardCode);
            break;

        case E2EXf_Profile::PROFILE22_CUSTOM:
            protect_ret = E2EXf_P22_CUSTOM(&Buffer[0], &BufferLength, nullptr, InputBufferLength, &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index),
                                           &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetProtectState(Index).P22ProtectState, ForwardCode, Profile22_custom_before, Profile22_custom_after);
            break;

        default:
            return ProtectResult::HardRuntimeError;
            break;
    }
    if (protect_ret == E_SAFETY_HARD_RUNTIMEERROR) return ProtectResult::HardRuntimeError;
    return ProtectResult::E_OK;
}

ProtectResult E2EXf_Forward_OutOfPlace(const E2EXf_Index& Index, const Payload& InputBuffer, Payload& Buffer, const std::uint32_t InputBufferLength, std::uint32_t& BufferLength,
                                       Std_TransformerForwardCode ForwardCode) {
    uint8_t protect_ret = E_SAFETY_HARD_RUNTIMEERROR;
    uint8_t* buffer = new uint8_t[InputBufferLength + hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index).headerLength];
    switch (hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index).Profile) {
        case E2EXf_Profile::PROFILE04:
            protect_ret = E2EXf_P04(buffer, &BufferLength, &InputBuffer[0], InputBufferLength, &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index),
                                    &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetProtectState(Index).P04ProtectState, ForwardCode);
            break;

        case E2EXf_Profile::PROFILE22:
            protect_ret = E2EXf_P22(buffer, &BufferLength, &InputBuffer[0], InputBufferLength, &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index),
                                    &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetProtectState(Index).P22ProtectState, ForwardCode);
            break;

        case E2EXf_Profile::PROFILE22_CUSTOM:
            protect_ret = E2EXf_P22_CUSTOM(buffer, &BufferLength, &InputBuffer[0], InputBufferLength, &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index),
                                           &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetProtectState(Index).P22ProtectState, ForwardCode, Profile22_custom_before, Profile22_custom_after);
            break;

        default:
            return ProtectResult::HardRuntimeError;
            break;
    }
    Buffer.clear();
    Buffer.insert(Buffer.begin(), buffer, buffer + BufferLength);
    delete[] buffer;
    if (protect_ret == E_SAFETY_HARD_RUNTIMEERROR) return ProtectResult::HardRuntimeError;
    return ProtectResult::E_OK;
}

CheckResult E2EXf_Check(const E2EXf_Index& Index, Payload& Buffer, const std::uint32_t InputBufferLength) {
    uint32_t BufferLength;
    uint8_t check_ret = E_SAFETY_HARD_RUNTIMEERROR;
    switch (hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index).Profile) {
        case E2EXf_Profile::PROFILE04:
            check_ret = E2EXf_Inv_P04(&Buffer[0], &BufferLength, nullptr, InputBufferLength, &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index),
                                      &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetCheckState(Index).P04CheckState, &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2ESMConfig(Index),
                                      &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetSMState(Index));
            break;

        case E2EXf_Profile::PROFILE22:
            check_ret = E2EXf_Inv_P22(&Buffer[0], &BufferLength, nullptr, InputBufferLength, &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index),
                                      &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetCheckState(Index).P22CheckState, &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2ESMConfig(Index),
                                      &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetSMState(Index));
            break;

        case E2EXf_Profile::PROFILE22_CUSTOM:
            check_ret = E2EXf_Inv_P22_CUSTOM(&Buffer[0], &BufferLength, nullptr, InputBufferLength, &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index),
                                             &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetCheckState(Index).P22CheckState, &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2ESMConfig(Index),
                                             &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetSMState(Index), Profile22_custom_before, Profile22_custom_after);
            break;

        default:
            break;
    }
    if (check_ret == E_SAFETY_HARD_RUNTIMEERROR) return CheckResult{E2E_P_ERROR, hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetSMState(Index).SMState};
    return CheckResult{hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetProfileCheckStatus(Index), hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetSMState(Index).SMState};
}

CheckResult E2EXf_Check_OutOfPlace(const E2EXf_Index& Index, const Payload& InputBuffer, Payload& Buffer, const std::uint32_t InputBufferLength, std::uint32_t& BufferLength) {
    uint8_t check_ret = E_SAFETY_HARD_RUNTIMEERROR;
    uint8_t* buffer = new uint8_t[InputBufferLength + hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index).headerLength];
    switch (hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index).Profile) {
        case E2EXf_Profile::PROFILE04:
            check_ret = E2EXf_Inv_P04(buffer, &BufferLength, &InputBuffer[0], InputBufferLength, &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index),
                                      &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetCheckState(Index).P04CheckState, &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2ESMConfig(Index),
                                      &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetSMState(Index));
            break;

        case E2EXf_Profile::PROFILE22:
            check_ret = E2EXf_Inv_P22(buffer, &BufferLength, &InputBuffer[0], InputBufferLength, &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index),
                                      &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetCheckState(Index).P22CheckState, &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2ESMConfig(Index),
                                      &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetSMState(Index));
            break;

        case E2EXf_Profile::PROFILE22_CUSTOM:
            check_ret = E2EXf_Inv_P22_CUSTOM(buffer, &BufferLength, &InputBuffer[0], InputBufferLength, &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2EConfig(Index),
                                             &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetCheckState(Index).P22CheckState, &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetE2ESMConfig(Index),
                                             &hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetSMState(Index), Profile22_custom_before, Profile22_custom_after);
        default:
            break;
    }
    Buffer.clear();
    Buffer.insert(Buffer.begin(), buffer, buffer + BufferLength);
    delete[] buffer;
    if (check_ret == E_SAFETY_HARD_RUNTIMEERROR) return CheckResult{E2E_P_ERROR, hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetSMState(Index).SMState};
    return CheckResult{hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetProfileCheckStatus(Index), hozon::netaos::e2e::E2EXf_Mapping::Instance()->GetSMState(Index).SMState};
}
}  // namespace e2e
}  // namespace netaos
}  // namespace hozon