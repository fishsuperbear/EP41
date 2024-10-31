#pragma once

#include <functional>
#include <memory>
#include <string>
#include <mutex>
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "logger.h"

namespace hozon {
namespace netaos {
namespace sensor {
class SensorProxyBase {
private:
    using WriteFun = std::function<int(std::string, std::shared_ptr<void>)>;
    WriteFun _func;
    std::recursive_mutex _func_mt;
public:
    explicit SensorProxyBase();
    virtual ~SensorProxyBase();

    virtual int Init() = 0;
    void RegisgerWriteFunc(WriteFun func);
    virtual void Deinit() = 0;
    virtual int Run() = 0;

// protected:
    template<typename T>
    int Write(std::string name, std::shared_ptr<T> data) {
        if(data == nullptr) {
            SENSOR_LOG_WARN << "Send " << name << " data is null";
            return -1;
        }
        std::shared_ptr<CmProtoBuf> idl_data = std::make_shared<CmProtoBuf>();
        idl_data->name(data->GetTypeName());
        std::string serialized_data;
        data->SerializeToString(&serialized_data);
        idl_data->str().assign(serialized_data.begin(), serialized_data.end());
        // SENSOR_LOG_INFO << "write " << name << " " <<  &_func;
        // std::lock_guard<std::recursive_mutex> lck(_func_mt);
        _func(name, idl_data);
        return 0;
    }
    
};
}   // namespace sensor
}   // namespace netaos 
}   // namespace hozon
