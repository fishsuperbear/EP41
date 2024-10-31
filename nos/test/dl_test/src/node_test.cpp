#include <memory>
#include <map>
#include <unistd.h>
#include "adf/include/log.h"
#include "inc/node_test.h"
#include "adf/include/class_register.h"

REGISTER_ADF_CLASS(Derived)
int32_t Derived::AlgProcess1(hozon::netaos::adf::NodeBundle* input,
                            const hozon::netaos::adf::ProfileToken& token) {
    NODE_LOG_INFO << "============================================================";
    std::unordered_map<std::string, std::vector<std::shared_ptr<hozon::netaos::adf::BaseData>>>& raw = input->GetRaw();
    for(auto it = raw.begin(); it != raw.end(); ++it) {
        NODE_LOG_INFO << "------------RECV: " << it->first;
    }
    return 0;
}
