#ifndef ARA_COM_INTERNAL_INSTANCE_BASE_H_
#define ARA_COM_INTERNAL_INSTANCE_BASE_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <map>
#include <memory>
#include <string>
#include <functional>
#include "ara/com/internal/manifest_config.h"
#include "ara/com/internal/internal_types.h"
#include "ara/com/serializer/someip_serializer.h"
#include "ara/com/serializer/someip_deserializer.h"
#include "ara/com/serializer/transformation.h"
#include "ara/core/result.h"

namespace ara {
namespace com {
namespace runtime {

struct TransportBindingConfig;
struct TransportServiceInterface;

class InstanceBase {
public:
    InstanceBase(const std::string& service_info);
    virtual ~InstanceBase();

    static void registerInitializeFunction(std::function<ara::core::Result<void>()> initialize_function);

    static void loadManifest(const std::string& service_info,
        const ara::com::runtime::ComServiceManifestConfig* manifest_config, const LoadConfigurationType& load_type = LoadConfigurationType_CPP);

    uint32_t methodIdx(const std::string& name);
    uint32_t eventIdx(const std::string& name);
    uint32_t fieldIdx(const std::string& name);

private:
    static void loadManifest(const ComServiceManifestConfig* config);

    static void parseManifest(char* manifest, std::vector<TransportBindingConfig>& binding_configs);

protected:
    std::string service_info_; // read-only
    std::shared_ptr<TransportServiceInterface> service_interface_; // read-only

    using Name = std::string;
    using Idx = uint32_t;
    std::map<Name, Idx> method_idxs_; // read-only
    std::map<Name, Idx> event_idxs_; // read-only
    std::map<Name, Idx> field_idxs_; // read-only
    std::map<Idx, Name> method_names_; // read-only
    std::map<Idx, Name> event_names_; // read-only
    std::map<Idx, Name> field_names_; // read-only

};

}  // namespace runtime
}  // namespace com
}  // namespace ara

#endif // ARA_COM_INTERNAL_INSTANCE_BASE_H_
/* EOF */
