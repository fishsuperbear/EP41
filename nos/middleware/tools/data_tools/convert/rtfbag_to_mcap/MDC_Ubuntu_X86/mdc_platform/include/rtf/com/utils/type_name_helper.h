/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: The template to check if the function GetTypeName() in data definiation is exist
 * Create: 2020-07-31
 */
#ifndef RTF_COM_UTILS_DATA_TYPE_HELPER_H
#define RTF_COM_UTILS_DATA_TYPE_HELPER_H

#include <string>
#include <type_traits>
#include <typeinfo>

#include "vrtf/vcc/utils/template_deduction_helper.h"
namespace ros {
namespace message_traits {
    template<class T>
    struct DataType;

    template<class T>
    struct IsMessage;
}
}
namespace rtf      {
namespace com      {
namespace utils    {
template<typename T>
struct HasFunctionValue {
public:
    template<typename U>
    static auto Check(int) -> decltype(ros::message_traits::IsMessage<U>::value, std::true_type());
    template<typename U>
    static std::false_type Check(...);
    static const bool value = std::is_same<decltype(Check<T>(0)), std::true_type>::value;
};


template<typename T>
typename std::enable_if<
    HasFunctionValue<typename std::decay<T>::type>::value && ros::message_traits::IsMessage<T>::value,
    std::string>::type GetTypeNameByRos(void)
{
    std::string name(ros::message_traits::DataType<typename std::decay<T>::type>::value());
    std::string headSymbol("/");
    return headSymbol + name;
}

template<typename T>
typename std::enable_if<
    HasFunctionValue<typename std::decay<T>::type>::value && !ros::message_traits::IsMessage<T>::value,
    std::string>::type GetTypeNameByRos(void) noexcept
{
    return "unknow";
}

template<typename T>
constexpr typename std::enable_if<
    !HasFunctionValue<typename std::decay<T>::type>::value, std::string>::type GetTypeNameByRos(void) noexcept
{
    return "unknow";
}

namespace TemplateDeduction = vrtf::vcc::utils::TemplateDeduction;
} // namespace utils
} // namespace com
} // namespace rtf

#endif
