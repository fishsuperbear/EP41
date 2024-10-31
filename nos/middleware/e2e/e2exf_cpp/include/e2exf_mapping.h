#ifndef E2EXf_Mapping_H_
#define E2EXf_Mapping_H_
#include <map>
#include <memory>
#include <mutex>
#include <utility>

#include "e2e/e2exf_cpp/include/e2exf_index.h"
namespace hozon {
namespace netaos {
namespace e2e {

class E2EXf_Mapping {
   public:
    static E2EXf_Mapping* Instance();

    void bind(const E2EXf_Index& index, const E2EXf_Config& config);

    const E2EXf_ConfigType& GetE2EConfig(const E2EXf_Index& index);

    const E2EXf_SMConfigType& GetE2ESMConfig(const E2EXf_Index& index);

    E2EXf_ProtectState& GetProtectState(const E2EXf_Index& index);

    E2EXf_CheckState& GetCheckState(const E2EXf_Index& index);

    E2EXf_SMState& GetSMState(const E2EXf_Index& index);

    E2E_PCheckStatusType GetProfileCheckStatus(const E2EXf_Index& index);

   private:
    E2EXf_Mapping() = default;
    ~E2EXf_Mapping() = default;

    static E2EXf_Mapping* instanceptr_;
    static std::mutex mtx_;
    std::map<E2EXf_Index, std::pair<E2EXf_Config, E2EXf_State>> indexmap_;
};
}  // namespace e2e
}  // namespace netaos
}  // namespace hozon
#endif