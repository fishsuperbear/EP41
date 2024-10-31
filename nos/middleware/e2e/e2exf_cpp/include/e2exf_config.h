#ifndef E2EXF_CONFIG_H_
#define E2EXF_CONFIG_H_

#include <cstdint>
#include <memory>
#include <vector>
#include <string>

#include "e2e/e2exf/include/e2exf.h"
#include "e2e/e2e/include/e2e_sm.h"

namespace hozon {
namespace netaos {
namespace e2e {

using Payload = std::vector<std::uint8_t>;
using E2EXf_SMConfigType = E2E_SMConfigType;
using E2EXf_SMStateType = E2E_SMStateType;
using E2EXf_PCheckStatusType = E2E_PCheckStatusType;

enum class ProtectResult : std::uint8_t { E_OK = 0U, HardRuntimeError = 0xFFU };

class CheckResult final {
   public:
    CheckResult(E2EXf_PCheckStatusType CheckStatus, E2EXf_SMStateType State) : checkstatus_(CheckStatus), smstate_(State) {}

    CheckResult() : CheckResult(E2EXf_PCheckStatusType::E2E_P_CHECKDISABLED, E2EXf_SMStateType::E2E_SM_DEINIT) {}

    ~CheckResult() = default;

    CheckResult(const CheckResult& ResultPre);

    CheckResult& operator=(const CheckResult& ResultPre) &;

    E2EXf_PCheckStatusType GetProfileCheckStatus() const { return checkstatus_; }

    E2EXf_SMStateType GetSMState() const { return smstate_; }

   private:
    E2EXf_PCheckStatusType checkstatus_;
    E2EXf_SMStateType smstate_;
};

class E2EXf_Config final {
   public:
    E2EXf_Config() = default;

    ~E2EXf_Config() = default;

    E2EXf_Config(const E2EXf_Config&) = default;

    E2EXf_Config& operator=(const E2EXf_Config&) & = default;

    explicit E2EXf_Config(const E2EXf_ConfigType& Config);

    E2EXf_Config(const E2EXf_ConfigType& Config, const E2EXf_SMConfigType& SMConfig) : config_(Config), smconfig_(SMConfig) {}

    const E2EXf_ConfigType& GetE2EXfConfig() const { return config_; }

    const E2EXf_SMConfigType& GetSMConfig() const { return smconfig_; }

    void SetE2EXfConfig(const E2EXf_ConfigType& Config) { config_ = Config; }

    void SetSMConfig(const E2EXf_SMConfigType& SMConfig) { smconfig_ = SMConfig; }

    bool Getis_set_sm() const;

   private:
    E2EXf_ConfigType config_{0, 0, true, false, false, noTransformerStatusForwarding};
    E2EXf_SMConfigType smconfig_{0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, false, false};

    bool is_set_sm_{false};
};
}  // namespace e2e
}  // namespace netaos
}  // namespace hozon

#endif