#pragma once

#include <unordered_map>
#include <functional>
#include <mutex>
#include "adf-lite/include/base.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

class BaseDataTypeMgr {
public:
    using CreateBaseDataTypeFunc = std::function<std::shared_ptr<BaseData>()>;

    static BaseDataTypeMgr& GetInstance() {
        static BaseDataTypeMgr instance;

        return instance;
    }

    std::shared_ptr<BaseData> Create(const std::string& name) {
        std::lock_guard<std::mutex> regist_lk(_regist_mutex);
        if (_func_map.find(name) != _func_map.end()) {
            return _func_map[name]();
        }

        return nullptr;
    }

    void Register(const std::string& name, CreateBaseDataTypeFunc func) {
        std::lock_guard<std::mutex> regist_lk(_regist_mutex);
        _func_map[name] = func;
    }

    uint32_t GetSize(const std::string& name) {
        std::lock_guard<std::mutex> regist_lk(_regist_mutex);
        if (_type_size_map.find(name) != _type_size_map.end()) {
            return _type_size_map[name];
        }
        return 0;
    }

    template<typename T>
    void SetSize(const std::string& name) {
        std::lock_guard<std::mutex> regist_lk(_regist_mutex);
        _type_size_map[name] = sizeof(T);
    }

private:
    BaseDataTypeMgr() {}
    ~BaseDataTypeMgr() {}
    
    std::unordered_map<std::string, CreateBaseDataTypeFunc> _func_map;
    std::unordered_map<std::string, uint32_t> _type_size_map;
    std::mutex _regist_mutex;
};

}
}
}

#define REGISTER_STRUCT_TYPE(name, type) { \
    hozon::netaos::adf_lite::BaseDataTypeMgr::GetInstance().Register(name, [](){ \
        return std::static_pointer_cast<hozon::netaos::adf_lite::BaseData>(std::make_shared<type>()); \
    }); \
    hozon::netaos::adf_lite::BaseDataTypeMgr::GetInstance().SetSize<type>(name); \
}
