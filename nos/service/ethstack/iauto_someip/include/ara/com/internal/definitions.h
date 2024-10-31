#ifndef ARA_COM_INTERNAL_DEFINITIONS_H_
#define ARA_COM_INTERNAL_DEFINITIONS_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <cstdint>
#include <type_traits>
#include <ara/core/string.h>

namespace ara {
namespace com {
namespace internal {

/**
 * @brief Definition of internal service ID type that should be binding agnostic.
 */
using ServiceId = ara::core::StringView;

/**
 * @brief Definition of internal instance ID type that should be binding agnostic.
 */
using InstanceId = std::uint16_t;

/**
 * @brief Definition of internal client ID type that should be binding agnostic.
 */
using ClientId = std::uint16_t;

/**
 * @brief Definition of internal client ID type that should be binding agnostic.
 */
using IpcClientId = std::uint64_t;

/**
 * @brief Definition of service version type
 */
using ServiceVersion = std::uint32_t;

/**
 * @brief Definition of service major version type
 */
using MajorVersion = std::uint32_t;

/**
 * @brief Definition of service minor version type
 */
using MinorVersion = std::uint32_t;

/**
 * @brief Definition of method id type
 */
using MethodId = std::uint16_t;

/**
 * @brief Definition of session id type
 */
using SessionId = std::uint32_t;

/**
 * @brief Definition of e2e data id
 */
using DataId = std::uint32_t;

/**
 * @brief Definition of e2e receive count
 */
using ReceivedCount = std::uint32_t;

static const InstanceId   INSTANCE_ID_UNKNOWN   = 0xFFFFU;      ///< "Null" instance ID
static const MajorVersion MAJOR_VERSION_UNKNOWN = 0xFFU;        ///< "Null" major version
static const MinorVersion MINOR_VERSION_UNKNOWN = 0xFFFFFFFFU;  ///< "Null" minor version


}  // namespace internal
}  // namespace com
}  // namespace ara

#endif  // ARA_COM_INTERNAL_DEFINITIONS_H_
/* EOF */
