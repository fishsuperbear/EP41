#ifndef ARA_COM_INTERNAL_MANIFEST_CONFIG_H
#define ARA_COM_INTERNAL_MANIFEST_CONFIG_H

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

namespace ara {
namespace com {
namespace runtime {

struct ComBindingManifestConfig {
    const char* name;
    bool fixed_prefix;
    const char* serialization; // someip dds ipc s2s
    const char* instances;
};

struct ComPortMappingConfig {
    const char* name;
    const char** instances;
};

struct ComServiceManifestConfig {
    const char* service;
    const char** methods;
    const char** events;
    const char** fields;
    const ComBindingManifestConfig* bindings;
    const ComPortMappingConfig* pport_mappings;
    const ComPortMappingConfig* rport_mappings;
};

}  // namespace runtime
}  // namespace com
}  // namespace ara

#endif // ARA_COM_INTERNAL_MANIFEST_CONFIG_H
/* EOF */
