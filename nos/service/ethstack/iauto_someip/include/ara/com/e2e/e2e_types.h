#ifndef INCLUDE_COM_E2E_E2E_TYPES_H_
#define INCLUDE_COM_E2E_E2E_TYPES_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include <functional>
#include <cstdint>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include "ne_someip_e2e.h"
#include "e2e_error_domain.h"
#include <functional>

/**
 * @brief Definition of e2e error code
 * [SWS_CM_90423]
 */
using Result = e2e::Result;

namespace ara {
namespace com {
namespace e2e {
inline namespace _19_11 {
/**
 * @brief Definition of e2e SMState
 * [SWS_CM_10474]
 */
using SMState = E2E_state_machine::E2EState;

/**
 * @brief Definition of e2e ProfileCheckStatus
 * [SWS_CM_90422]
 */
using ProfileCheckStatus = E2E_state_machine::E2ECheckStatus;

///**
// * @brief Definition of e2e error code
// * [SWS_CM_90423]
// */
//using Result = e2e::Result;

/**
 * @brief Definition of e2e error code
 * [SWS_CM_10474]
 */
using E2EErrorCode = ara::com::e2e::E2EError;

/**
 * @brief Definition of e2e message count
 */
using MessageCounter = std::uint32_t;

/**
 * @brief Definition of e2e DataId
 */
using DataID = std::uint32_t;

/**
 * @brief Definition of e2e error handler
 *
 * @uptrace{SWS_CM_10470}
 */
using E2EErrorHandler = std::function<void(ara::com::e2e::E2EErrorCode   errorCode,
                                           ara::com::e2e::DataID         dataID,
                                           ara::com::e2e::MessageCounter messageCounter)>;

}  // inline namespace _19_11
}  // namespace e2e
}  // namespace com
}  // namespace ara

#endif  // INCLUDE_COM_E2E_E2E_TYPES_H_
/* EOF */
