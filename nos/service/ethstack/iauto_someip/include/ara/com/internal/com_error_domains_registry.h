#ifndef ARA_COM_INTERNAL_COM_ERROR_DOMAINS_REGISTRY_H_
#define ARA_COM_INTERNAL_COM_ERROR_DOMAINS_REGISTRY_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include "ara/core/error_domain.h"
#include "ara/core/map.h"

namespace ara {
namespace com {
namespace internal {
namespace error_domains {

/**
 * @brief Definition of Registry.
 *
 */
class Registry {
public:
    static Registry* instance();

    /**
     * @brief Save Error Domain
     *
     * @param domain ErrorDomain
     */
    void registerDomain(const ara::core::ErrorDomain& domain);

    /**
     * @brief Get the Domain object
     *
     * @param id Error Doamin ID
     * @return const ara::core::ErrorDomain
     */
    const ara::core::ErrorDomain& getDomain(const ara::core::ErrorDomain::IdType& id);

private:
    using domain_id  = ara::core::ErrorDomain::IdType;
    using domain_ref = std::reference_wrapper<const ara::core::ErrorDomain>;
    ara::core::Map<domain_id, domain_ref> known_domains;
};

}  // namespace error_domains
}  // namespace internal
}  // namespace com
}  // namespace ara

#endif  // ARA_COM_INTERNAL_COM_ERROR_DOMAINS_REGISTRY_H_
/* EOF */
