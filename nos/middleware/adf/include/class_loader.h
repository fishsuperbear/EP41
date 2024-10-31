#pragma once
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "adf/include/log.h"

namespace hozon {
namespace netaos {
namespace adf {

template <class BaseClass>
class ClassLoader {
   public:
    ClassLoader() {}

    ~ClassLoader() {}

    using CreateClassFunc = std::function<std::shared_ptr<BaseClass>()>;

    template <typename DerivedClass>
    void RegisterClass(const std::string& class_name, CreateClassFunc func) {
        std::lock_guard<std::mutex> lck(_mtx);
        // ADF_EARLY_LOG << "Register class " << class_name << " in " << this;
        _factory_map[class_name] = func;
    }

    std::shared_ptr<BaseClass> Create(const std::string& class_name) {
        std::lock_guard<std::mutex> lck(_mtx);
        if (_factory_map.find(class_name) == _factory_map.end()) {
            return nullptr;
        }
        return (_factory_map[class_name]());
    }

    void UnRegisterClass(const std::string& class_name) {
        std::lock_guard<std::mutex> lck(_mtx);
        if (_factory_map.find(class_name) != _factory_map.end()) {
            // ADF_EARLY_LOG << "delete from factory map: " << class_name;
            _factory_map.erase(class_name);
        }
    }

   private:
    std::mutex _mtx;
    std::unordered_map<std::string, CreateClassFunc> _factory_map;
};

}  // namespace adf
}  // namespace netaos
}  // namespace hozon
