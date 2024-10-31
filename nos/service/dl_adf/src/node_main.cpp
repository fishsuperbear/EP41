#include <functional>
#include "dl_adf/include/library_loader.h"
#include "dl_adf/include/dl_config.h"
#include "dl_adf/include/log.h"
#include "adf/include/class_register.h"

using namespace hozon::netaos::dl_adf;


int main(int argc, char* argv[]) {
    if(argc != 2) {
        DL_EARLY_LOG << "Adf main argument number: " << argc << " != 2";
        return -1;
    }
    std::string yaml_path = argv[1];
    DL_EARLY_LOG << "Adf main yaml path:" << yaml_path;

    
    NodeConfig _config;
    _config.Parse(yaml_path);
    LibraryLoader loader;
    if(0 != loader.Load(_config.module.path)) {
        DL_EARLY_LOG << "Load path: " << _config.module.path << " fail.";
        return -1;
    }
        
    std::shared_ptr<hozon::netaos::adf::NodeBase> node = hozon::netaos::adf::g_class_loader.Create(_config.module.class_name);
    if(node == nullptr) {
        DL_EARLY_LOG << "Fail to creat class " << _config.module.class_name;
        return -1;
    }
    
    node->Start(yaml_path);

    node->NeedStopBlocking();
    node->Stop();
    hozon::netaos::adf::g_class_loader.UnRegisterClass(_config.module.class_name);
    
    // loader.UnloadAll();
    return 0;
}

