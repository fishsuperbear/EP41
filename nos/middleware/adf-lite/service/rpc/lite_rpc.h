#pragma once
#include <functional>
#include <unordered_map>
#include "adf-lite/include/base.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

class LiteRpc {
public:
    static LiteRpc& GetInstance() {
        static LiteRpc instance;

        return instance;
    }
    using ServiceFunc = std::function<int32_t(BaseDataTypePtr& input)>;
    using StringServiceFunc = std::function<int32_t(std::string& value)>;

    void RegisterServiceFunc(const std::string& func_name, ServiceFunc func);
    ServiceFunc GetServiceFunc(const std::string& func_name);

    void RegisterStringServiceFunc(const std::string& func_name, StringServiceFunc func);
    StringServiceFunc GetStringServiceFunc(const std::string& func_name);
private:
    LiteRpc();
    ~LiteRpc();
    std::unordered_map<std::string, ServiceFunc> rpc_func_map;
    std::unordered_map<std::string, StringServiceFunc> rpc_string_func_map;
    std::mutex _regist_rpc_func_map_mutex;
    std::mutex _regist_rpc_string_func_map_mutex;
};


}    
}
}