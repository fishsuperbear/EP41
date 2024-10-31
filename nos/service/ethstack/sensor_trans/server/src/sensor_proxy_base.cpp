#include "sensor_proxy_base.h"


namespace hozon {
namespace netaos {
namespace sensor {
SensorProxyBase::SensorProxyBase() {}
SensorProxyBase::~SensorProxyBase() {}
void SensorProxyBase::RegisgerWriteFunc(WriteFun func) {
    _func = func;
}

// template<typename T>
// int SensorProxyBase::Write(std::string name, std::shared_ptr<T> data) {
//     std::shared_ptr<CmProtoBuf> idl_data = std::make_shared<CmProtoBuf>();
//     idl_data->name(data->GetTypeName);
//     std::string serialized_data;
//     data->SerializeToString(&serialized_data);
//     idl_data->str().assign(serialized_data.begin(), serialized_data.end());

//     _func(name, idl_data);
//     return 0;
// }

}   // namespace sensor
}   // namespace netaos
}   // namespace hozon