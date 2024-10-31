/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: Define types in rtftools
 * Create: 2020-10-24
 */
#ifndef RTF_TOOLS_TYPES_H
#define RTF_TOOLS_TYPES_H

#include "ara/core/map.h"
#include "ara/core/string.h"
#include "ara/core/vector.h"

#ifdef AOS_TAINT
#ifndef COVERITY_TAINT_SET_DEFINITION
#define COVERITY_TAINT_SET_DEFINITION
    /**
    * @brief Function for Stain Modeling
    * @details The function is used only when the compilation macro AOS_TAINT is enabled.
    */
    static void Coverity_Tainted_Set(void *buf){}
#endif
#endif

namespace rtf {
namespace rtfevent {
/**
 * @brief the class to store basic event info
 * @example Name Example <br>
 * EventName = ${EventType}[${InstanceShortname}] <br>
 * @note instanceShortName_ could be empty
 */
class RtfEventInfo {
public:
    ara::core::Vector<ara::core::String> GetSubs() const
    {
        return subList_;
    }
    void SetSubs(ara::core::Vector<ara::core::String> subs)
    {
#ifdef AOS_TAINT
        Coverity_Tainted_Set((void *)(&subs));
#endif
        subList_.swap(subs);
    }
    ara::core::String GetPub() const
    {
        return pub_;
    }
    void SetPub(const ara::core::String pub)
    {
#ifdef AOS_TAINT
        Coverity_Tainted_Set((void *)(&pub));
#endif
        pub_ = pub;
    }
    ara::core::String GetEventName() const
    {
        auto res = eventType_;
        if (!instanceShortName_.empty()) {
            res += '[' + instanceShortName_ + ']';
        }
        return res;
    }
    void SetEventName(const ara::core::String eventName)
    {
#ifdef AOS_TAINT
        Coverity_Tainted_Set((void *)(&eventName));
#endif
        ara::core::String eventNameTemp = eventName;
        ara::core::String instanceShortNameTemp = "";
        auto leftBracket = eventName.find_first_of('[');
        auto rightBracket = eventName.find_last_of(']');
        if (leftBracket != ara::core::String::npos && rightBracket != ara::core::String::npos &&
            leftBracket < rightBracket) {
                eventNameTemp = eventName.substr(0, leftBracket);
                instanceShortNameTemp = eventName.substr(leftBracket + 1UL, (rightBracket - leftBracket) - 1UL);
        }
        eventType_ = eventNameTemp;
        instanceShortName_ = instanceShortNameTemp;
    }
    const ara::core::String& GetInstanceShortName() const
    {
        return instanceShortName_;
    }
    void SetInstanceShortName(ara::core::String instanceName)
    {
#ifdef AOS_TAINT
        Coverity_Tainted_Set((void *)(&instanceName));
#endif
        instanceShortName_ = std::move(instanceName);
    }
    ara::core::String GetEventType() const
    {
        return eventType_;
    }
    void SetEventType(const ara::core::String type)
    {
#ifdef AOS_TAINT
        Coverity_Tainted_Set((void *)(&type));
#endif
        eventType_ = type;
    }
    void SetAttribute(const ara::core::Map<ara::core::String, ara::core::String>& attributeList)
    {
#ifdef AOS_TAINT
        Coverity_Tainted_Set((void *)(&attributeList));
#endif
        attributeList_ = attributeList;
    }
    ara::core::Map<ara::core::String, ara::core::String> GetAttribute() const
    {
        return attributeList_;
    }

    ara::core::String GetInstanceId() const { return instanceId_; }

    void SetInstanceId(const ara::core::String &instanceId) { instanceId_ = instanceId; }

private:
    ara::core::String pub_;
    ara::core::Vector<ara::core::String> subList_;
    ara::core::String eventType_;
    /** instance name is surrounded by '[' ']' */
    ara::core::String instanceShortName_;
    ara::core::Map<ara::core::String, ara::core::String> attributeList_;
    ara::core::String instanceId_;
};

class EventFilter {
public:
    enum class Type: uint8_t {
        PUBLISHER,
        SUBSCRIBER,
        ALL
    };
    enum class CommunicableType: uint8_t {
        UNDEFINED,
        LOCAL,
        NETWORK,
        CROSS,
        ALL
    };
    explicit EventFilter(const Type &type)
        : communicableType_(CommunicableType::UNDEFINED)
    {
#ifdef AOS_TAINT
        Coverity_Tainted_Set((void *)(&type));
#endif
        filterType_ = type;
    }
    ~EventFilter() = default;
    void SetType(const Type &type)
    {
#ifdef AOS_TAINT
        Coverity_Tainted_Set((void *)(&type));
#endif
        filterType_ = type;
    }
    Type GetType() const
    {
        return filterType_;
    }
    void SetCommunicableType(const CommunicableType& type)
    {
        communicableType_ = type;
    }
    CommunicableType GetCommunicableType() const
    {
        return communicableType_;
    }
private:
    Type filterType_;
    CommunicableType communicableType_;
};
}
}
#endif
