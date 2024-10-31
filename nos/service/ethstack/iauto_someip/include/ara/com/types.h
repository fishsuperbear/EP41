#ifndef INCLUDE_COM_TYPES_H_
#define INCLUDE_COM_TYPES_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <memory>
#include <functional>
#include <ara/core/vector.h>
#include "ara/com/instance_identifier.h"
#include "ara/com/sample_ptr.h"

namespace ara {
namespace com {

namespace runtime {
class ProxyInstance;
class SkeletonInstance;
struct TransportBindingInstanceInfo;
} // namespace runtime

inline namespace _19_11 {

class ProxyMemberBase {
public:
    ProxyMemberBase(ara::core::string name, const std::shared_ptr<ara::com::runtime::ProxyInstance> instance, uint32_t idx)
    : name_(name)
    , instance_(instance)
    , idx_(idx)
    {
    }

protected:
    ara::core::string name_;
    std::shared_ptr<ara::com::runtime::ProxyInstance> instance_;
    uint32_t idx_;
};

class SkeletonMemberBase {
public:
    SkeletonMemberBase(ara::core::string name, const std::shared_ptr<ara::com::runtime::SkeletonInstance> instance, uint32_t idx)
    : name_(name)
    , instance_(instance)
    , idx_(idx)
    {
    }

protected:
    ara::core::string name_;
    std::shared_ptr<ara::com::runtime::SkeletonInstance> instance_;
    uint32_t idx_;
};

/**
 * @brief Instance Identifier Container Class
 *
 * @uptrace{SWS_CM_00319}
 */
using InstanceIdentifierContainer = ara::core::Vector<InstanceIdentifier>;
using InstanceIdContainer = ara::core::Vector<internal::InstanceId>;

/**
 * @brief Method call processing modes for the service implementation side (skeleton).
 *
 * @uptrace{SWS_CM_00301}
 */
enum class MethodCallProcessingMode : uint8_t {
    kPoll,
    kEvent,
    kEventSingleThread
};

/**
 * @brief Identifier for a triggered FindService request.
 *
 * This identifier is needed to later cancel the FindService request.
 *
 * @uptrace{SWS_CM_00303}
 */
#if 0
struct FindServiceHandle {
    internal::ServiceId    service_id;
    ara::com::InstanceIdContainer   instance_id;
    std::uint32_t          uid;

    bool operator==( const FindServiceHandle& other ) const {
        if ( service_id != other.service_id ||
             uid != other.uid ) {
            return false;
        } else {
            return true;
        }
    }

    bool operator<( const FindServiceHandle& other ) const {
        return ( uid < other.uid );
    }
};
#endif

struct FindServiceHandle {
public:
    FindServiceHandle();
    ~FindServiceHandle();
    FindServiceHandle(const FindServiceHandle& other);
    FindServiceHandle& operator=(const FindServiceHandle& other);
    bool operator==(const FindServiceHandle& other) const ;
    bool operator<(const FindServiceHandle& other) const ;
    friend class ara::com::runtime::ProxyInstance;
private:
    explicit FindServiceHandle(bool internal);
    uint64_t uuid_;
};

class HandleType {
public:
    HandleType(const HandleType& other) = default;
    HandleType& operator=(const HandleType& other) = default;
    bool operator==(const HandleType& other);
    bool operator<(const HandleType& other);
    const ara::com::InstanceIdentifier GetInstanceId() const;
    friend class ara::com::runtime::ProxyInstance;
    friend class ara::com::runtime::SkeletonInstance;
private:
    explicit HandleType(const std::shared_ptr<ara::com::runtime::TransportBindingInstanceInfo> instance_info);
    std::shared_ptr<ara::com::runtime::TransportBindingInstanceInfo> instance_info_;
};


/**
 * Container for a list of service handles.
 *
 * @see ara::com::FindService
 *
 * @uptrace{SWS_CM_00304}
 */
template <typename Handle>
using ServiceHandleContainer = ara::core::Vector<Handle>;

/**
 * @brief Pointer to allocated sample on service side.
 *
 * @uptrace{SWS_CM_00308}
 */
template <typename T>
using SampleAllocateePtr = std::unique_ptr<T>;

/**
 * @brief Receive handler method, which is semantically a void(void) function.
 *
 * @uptrace{SWS_CM_00309}
 */
using EventReceiveHandler = std::function<void()>;

/**
 * @brief Definition of the subscription state of an Event.
 *
 * @uptrace{SWS_CM_00310}
 */
enum class SubscriptionState : uint8_t {
    kSubscribed,
    kNotSubscribed,
    kSubscriptionPending
};

/**
 * @brief Definition of the subscription state of an Event.
 *
 * @uptrace{SWS_CM_00311}
 */
using SubscriptionStateChangeHandler = std::function<void(SubscriptionState)>;

/**
 * @brief Handler that gets called in case service availability for services which have been
 * searched for via FindService() has changed.
 *
 * @uptrace{SWS_CM_00383}
 */
template <typename T>
using FindServiceHandler = std::function<void(ServiceHandleContainer<T>, FindServiceHandle)>;


}  // inline namespace _19_11
}  // namespace com
}  // namespace ara

#endif  // INCLUDE_COM_TYPES_H_
/* EOF */
