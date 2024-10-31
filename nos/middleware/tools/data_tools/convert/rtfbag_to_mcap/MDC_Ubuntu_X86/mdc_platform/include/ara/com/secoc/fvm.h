/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
 * Description: Fvm library api class in CM.
 * Create: 2021-11-07
 */
#ifndef ARA_COM_SECOC_FVM_H
#define ARA_COM_SECOC_FVM_H
#include <memory>
#include "ara/core/array.h"
#include "ara/com/secoc/fvm_error_domain.h"
namespace ara {
namespace com {
namespace secoc {
constexpr std::size_t FV_MAX_LENGTH_BYTE = 8U;
struct FVContainer {
    std::uint64_t length;   // bits
    ara::core::Array<std::uint8_t, FV_MAX_LENGTH_BYTE> value;
};

class FVMImpl;
class FVM {
public:
    FVM();
    ~FVM() {};
    ara::core::Result<void, SecOcFvmErrc> Initialize() noexcept;
    ara::core::Result<FVContainer, SecOcFvmErrc> GetRxFreshness(std::uint16_t SecOCFreshnessValueID,
                                                                const FVContainer &SecOCTruncatedFreshnessValue,
                                                                std::uint16_t SecOCAuthVerifyAttempts) const noexcept;
    ara::core::Result<FVContainer, SecOcFvmErrc> GetTxFreshness(std::uint16_t SecOCFreshnessValueID) const noexcept;
private:
    std::shared_ptr<FVMImpl> fvmImpl_;
};
}
}
}
#endif  // ARA_COM_SECOC_FVM_H
