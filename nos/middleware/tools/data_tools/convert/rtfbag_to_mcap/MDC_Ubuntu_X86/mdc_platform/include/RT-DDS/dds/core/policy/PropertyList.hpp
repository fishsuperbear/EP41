/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: PropertyList.hpp
 */

#ifndef RT_DDS_PROPERTYLIST_HPP
#define RT_DDS_PROPERTYLIST_HPP

#include <unordered_map>

namespace dds {
namespace core {
namespace policy {
static const std::string MAX_LATENCYTIMEOUT= "4294967295";
/**
 * @brief A Set of Property
 */
class PropertyList {
public:
    /**
    * @ingroup dds::core::policy
    * @brief PropertyList Constructor
    * @par Description
    * 1. FUNCTION_Description
    * ...
    * @param[in] NONE
    * ...
    * @return constructor
    * ...
    * @req{AR-iAOS-RCS-DDS-10050, AR20220630142735
    * NAME_OF_THE_RELATED_AR_NUMBER.,
    * AR-RCS-DDS-RTPS-00018
    * }
    */
    PropertyList() = default;

    /**
    * @ingroup dds::core::policy
    * @brief PropertyList Destructor
    * @par Description
    * 1. FUNCTION_Description
    * ...
    * @param[in] NONE
    * ...
    * @return Destructor
    * ...
    * @req{AR-iAOS-RCS-DDS-10050, AR20220630142735
    * NAME_OF_THE_RELATED_AR_NUMBER.,
    * AR-RCS-DDS-RTPS-00018
    * }
    */
    ~PropertyList() = default;

    /**
    * @ingroup dds::core::policy
    * @brief Propertylist Setter
    * @par Description
    * 1. FUNCTION_Description
    * ...
    * @param[in] propertyListKind an enum to specify which property to be set
    * @param[in] propertyVal property value to be set
    * ...
    * @return void
    * ...
    * @req{AR-iAOS-RCS-DDS-10050, AR20220630142735
    * NAME_OF_THE_RELATED_AR_NUMBER.,
    * AR-RCS-DDS-RTPS-00018
    * }
    */
    void SetProperty(std::string propertyListKind, std::string propertyVal)
    {
        propertyListMap_[propertyListKind] = propertyVal;
    }

    /**
    * @ingroup dds::core::policy
    * @brief PropertyList Getter
    * @par Description
    * 1. FUNCTION_Description
    * ...
    * @param[in] propertyListKind an enum to specify which property to be got
    * ...
    * @return the value of correspoding property
    * ...
    * @req{AR-iAOS-RCS-DDS-10050, AR20220630142735
    * NAME_OF_THE_RELATED_AR_NUMBER.,
    * AR-RCS-DDS-RTPS-00018
    * }
    */
    std::string GetProperty(std::string propertyListKind) const
    {
        if (propertyListMap_.find(propertyListKind) != propertyListMap_.end()) {
            return propertyListMap_.at(propertyListKind);
        } else {
            return "";
        }
    }

    /**
    * @ingroup dds::core::policy
    * @brief Enable Latency Status value
    * @par Description
    * 1. FUNCTION_Description
    * ...
    * @param[in] NONE
    * ...
    * @return return 1U, representing LatencyStat is enabled
    * ...
    * @req{AR-iAOS-RCS-DDS-10050, AR20220630142735
    * NAME_OF_THE_RELATED_AR_NUMBER.,
    * AR-RCS-DDS-RTPS-00018
    * }
    */
    static std::string EnableLatencyStat()
    {
        return "1";
    }

    /**
    * @ingroup dds::core::policy
    * @brief disable Latency Status value
    * @par Description
    * 1. FUNCTION_Description
    * ...
    * @param[in] NONE
    * ...
    * @return return 0U, representing LatencyStat is disabled
    * ...
    * @req{AR-iAOS-RCS-DDS-10050, AR20220630142735
    * NAME_OF_THE_RELATED_AR_NUMBER.,
    * AR-RCS-DDS-RTPS-00018
    * }
    */
    static std::string DisableLatencyStat()
    {
        return "0";
    }

    /**
    * @ingroup dds::core::policy
    * @brief Enable Latency Status value
    * @param[in] timeOut the latency of TimeOut user set
    * @return return the string of timeOut
    */
    static std::string SetLatencyTimeOut(uint32_t timeOut)
    {
        return std::to_string(timeOut);
    }

    std::unordered_map<std::string, std::string> GetPropertyListMap() const noexcept
    {
        return propertyListMap_;
    }

private:
    std::unordered_map<std::string, std::string> propertyListMap_{{"LATENCY_STAT", "0"},
                                                                  {"LATENCY_TIMEOUT", MAX_LATENCYTIMEOUT}};
};

}
}
}
#endif /* RT_DDS_PROPERTYLIST_HPP */
