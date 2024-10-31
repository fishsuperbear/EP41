#ifndef ARA_COM_INTERNAL_CONSTRUCTION_TOKEN_H_
#define ARA_COM_INTERNAL_CONSTRUCTION_TOKEN_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include "ara/com/types.h"
#include "ara/com/internal/definitions.h"

namespace ara {
namespace com {
namespace internal {

/**
 * @brief Definition of ConstructionToken.
 *
 */
class ConstructionToken {
   public:
    /**
     * @brief Construct a new Construction Token object.
     *
     * @param service_id Service id.
     * @param instance_id Instance id.
     * @param mode Method processing mode.
     */
    ConstructionToken( ara::com::internal::ServiceId      service_id,
                       ara::com::InstanceIdentifierContainer       instance_id,
                       ara::com::MethodCallProcessingMode mode )
        : service_id_( service_id ), instance_id_( instance_id ), mode_( mode ) {}
    ConstructionToken( ara::com::internal::ServiceId      service_id,
                       ara::com::InstanceIdentifier       instance_id,
                       ara::com::MethodCallProcessingMode mode )
        : service_id_( service_id ), mode_( mode ) {
            instance_id_.push_back(instance_id);
        }

    /* @uptrace{SWS_CM_10433} */
    ConstructionToken( ConstructionToken&& token ) = default;
    ConstructionToken& operator=( ConstructionToken&& token ) = default;
    ConstructionToken( const ConstructionToken& ) = delete;
    ConstructionToken& operator=( const ConstructionToken& ) = delete;

    /**
     * @brief Get the Service Id object.
     *
     * @return ServiceId Service id.
     */
    ServiceId GetServiceId() { return service_id_; }

    /**
     * @brief Get the Instance Identifier object.
     *
     * @return InstanceIdentifier Instance identifier.
     */
    ara::com::InstanceIdentifierContainer GetInstanceIdentifier() { return instance_id_; }

    /**
     * @brief Get the Method Call Mode object.
     *
     * @return MethodCallProcessingMode Method processing mode.
     */
    MethodCallProcessingMode GetMethodCallMode() { return mode_; }

   private:
    ara::com::internal::ServiceId      service_id_;
    ara::com::InstanceIdentifierContainer       instance_id_;
    ara::com::MethodCallProcessingMode mode_;
};

}  // namespace internal
}  // namespace com
}  // namespace ara

#endif  // ARA_COM_INTERNAL_CONSTRUCTION_TOKEN_H_
/* EOF */
