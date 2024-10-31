#include "dl_adf/include/dl_config.h"


namespace hozon {
namespace netaos {
namespace dl_adf {
int32_t NodeConfig::Parse(const std::string& file) {
    _node = YAML::LoadFile(file);
    if(!_node) {
        DL_EARLY_LOG << "Fail to load config file: " << file;
    }
    DO_OR_ERROR(_node["module"], "module");
    DO_OR_ERROR(_node["module"]["className"], "module.className");
    DO_OR_ERROR(_node["module"]["path"], "module.path");
    module.class_name = _node["module"]["className"].as<std::string>();
    module.path = _node["module"]["path"].as<std::string>();
    return 0;
}

}   // namespace dl_adf
}   // namespace netaos 
}   // namespace hozon