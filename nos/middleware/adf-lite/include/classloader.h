#pragma once

#include <string>
#include <functional>
#include <unordered_map>
#include <mutex>
#include <iostream>
#include <memory>
#include <dlfcn.h>
#include "adf-lite/include/adf_lite_internal_logger.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

class LibraryLoader {
public:
    int32_t Load(const std::string& lib_path);
    void Unload();

    void* GetSymbol(const std::string& symbol_name);

private:
    void* _handle = nullptr;
    std::string _lib_path;
};

template<typename BaseClass>
class ClassLoader {
public:
    using CreateClassFunc = std::function<std::shared_ptr<BaseClass>()>;

    template<typename DerivedClass>
    void RegisterClass(const std::string& class_name, CreateClassFunc func) {
        std::lock_guard<std::mutex> lk(_mtx);
        _factory_map[class_name] = func;
        ADF_INTERNAL_LOG_DEBUG << "RegisterClass: [" << class_name << "] Success";
    }

    std::shared_ptr<BaseClass> Create(const std::string& class_name) {
        std::lock_guard<std::mutex> lk(_mtx);

        if (_factory_map.find(class_name) == _factory_map.end()) {
            ADF_INTERNAL_LOG_ERROR << "class_name: [" << class_name << "] not Registered";
            return nullptr;
        }

        return _factory_map[class_name]();
    }

private:
    std::mutex _mtx;
    std::unordered_map<std::string, CreateClassFunc> _factory_map;
};

}
}
}

