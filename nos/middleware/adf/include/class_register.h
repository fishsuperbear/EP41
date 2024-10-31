#pragma once
#include "adf/include/class_loader.h"
#include "adf/include/node_base.h"

namespace hozon {
namespace netaos {
namespace adf {

extern ClassLoader<hozon::netaos::adf::NodeBase> g_class_loader;

#define REGISTER_DERIVED_CLASS(type, uniqueid)                                                           \
    class TypeInstance##uniqueid {                                                                       \
       public:                                                                                           \
        TypeInstance##uniqueid() {                                                                       \
            hozon::netaos::adf::g_class_loader.RegisterClass<type>(#type, []() {                         \
                return std::static_pointer_cast<hozon::netaos::adf::NodeBase>(std::make_shared<type>()); \
            });                                                                                          \
        }                                                                                                \
        ~TypeInstance##uniqueid() {}                                                                     \
    };                                                                                                   \
    static TypeInstance##uniqueid g_type_instance_##uniqueid;

#define REGISTER_ADF_CLASS_INTERNAL_1(type, uniqueid) REGISTER_DERIVED_CLASS(type, uniqueid)

#define REGISTER_ADF_CLASS(type) REGISTER_ADF_CLASS_INTERNAL_1(type, __COUNTER__)

}  // namespace adf
}  // namespace netaos
}  // namespace hozon
