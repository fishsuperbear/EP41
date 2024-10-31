/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: The Phm Error Code.
 * Create: 2020-11-04
 */

#ifndef VRTF_PHM_ERROR_DOMAIN_H
#define VRTF_PHM_ERROR_DOMAIN_H

#include <map>
#include "ara/core/error_domain.h"
#include "ara/core/error_code.h"
#include "ara/core/exception.h"
namespace ara  {
namespace phm {
enum class PhmErrc : ara::core::ErrorDomain::CodeType {
    kGeneralError = 1U,
    kInvalidArguments = 2U,
    kCommunicationError = 3U,
};
/**
 * @defgroup DetectorProxy DetectorProxy
 * @brief Container for all PhmErrorDomain objects.
 * @ingroup DetectorProxy
 */
/* AXIVION Next Line AutosarC++19_03-M3.4.1 : standard class definition, can't define in function [SWS_CORE_00611] */
class PhmException : public ara::core::Exception {
public:
    using Exception::Exception;
    ~PhmException(void) override = default;
    PhmException(PhmException const &) = default;
    PhmException(PhmException &&) = default;
    PhmException &operator=(PhmException const &) & = delete;
    PhmException &operator=(PhmException &&) & = delete;
};

class PhmErrorDomain final : public ara::core::ErrorDomain {
public:
    /**
     * @ingroup DetectorProxy
     * @brief Constructor of PhmErrorDomain.
     * @return PhmErrorDomain
     * @req{AR-iAOS-RTF-RTFPHM-00047,
     * RTFPHM allows the client to report faults to a specified PHM instance.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00047
     * }
     */
    constexpr PhmErrorDomain() noexcept : ErrorDomain(PhmErrcDomainId) {}
    /**
     * @ingroup DetectorProxy
     * @brief Destructor of PhmErrorDomain.
     * @req{AR-iAOS-RTF-RTFPHM-00047,
     * RTFPHM allows the client to report faults to a specified PHM instance.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00047
     * }
     */
    ~PhmErrorDomain() = default;
    PhmErrorDomain(PhmErrorDomain const &) = delete;
    PhmErrorDomain(PhmErrorDomain &&) = delete;
    PhmErrorDomain &operator=(PhmErrorDomain const &) & = delete;
    PhmErrorDomain &operator=(PhmErrorDomain &&) & = delete;
    /**
     * @ingroup DetectorProxy
     * @brief Get the name.
     * @return char const *
     * @req{AR-iAOS-RTF-RTFPHM-00047,
     * RTFPHM allows the client to report faults to a specified PHM instance.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00047
     * }
     */
    /* Axivion Next Line AutosarC++19_03-A27.0.4, AutosarC++19_03-A3.9.1: char const * is defined by [SWS_CORE_00152] */
    char const *Name() const noexcept final
    {
        /* AXIVION Next Line AutosarC++19_03-A5.1.1 : model name is phm */
        return "Phm";
    }
    /**
     * @ingroup DetectorProxy
     * @brief Get the error message.
     * @param[in] errorCode The error code.
     * @return char const *
     * @req{AR-iAOS-RTF-RTFPHM-00047,
     * RTFPHM allows the client to report faults to a specified PHM instance.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00047
     * }
     */
    /* Axivion Next Line AutosarC++19_03-A27.0.4, AutosarC++19_03-A3.9.1: char const * is defined by [SWS_CORE_00153] */
    char const *Message(ara::core::ErrorDomain::CodeType errorCode) const noexcept final
    {
    /* Axivion Next Line AutosarC++19_03-A27.0.4, AutosarC++19_03-A3.9.1: char const * is defined by [SWS_CORE_00153] */
        std::map<ara::core::ErrorDomain::CodeType, char const *> mapCode {
            {ara::core::ErrorDomain::CodeType(PhmErrc::kGeneralError),       "Some unspecified error occurred"},
            {ara::core::ErrorDomain::CodeType(PhmErrc::kInvalidArguments),   "Invalid argument was passed"},
            {ara::core::ErrorDomain::CodeType(PhmErrc::kCommunicationError), "Communication error occurred"}
        };
        if (mapCode.find(errorCode) == mapCode.end()) {
            return "";
        }
        return mapCode[errorCode];
    }
    /**
     * @ingroup DetectorProxy
     * @brief Throw error as exception.
     * @param[in] errorCode The error code.
     * @return char const *
     * @req{AR-iAOS-RTF-RTFPHM-00047,
     * RTFPHM allows the client to report faults to a specified PHM instance.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00047
     * }
     */
    void ThrowAsException(ara::core::ErrorCode const &errorCode) const noexcept(false) final
    {
        ara::core::ThrowOrTerminate<PhmException>(errorCode);
    }
private:
    constexpr static ErrorDomain::IdType PhmErrcDomainId {0x8000000000000399ULL}; // temp phm id
};

constexpr PhmErrorDomain g_PhmErrorDomain;
/**
 * @ingroup DetectorProxy
 * @brief Get the phm error domain.
 * @return ara::core::ErrorDomain
 * @req{AR-iAOS-RTF-RTFPHM-00047,
 * RTFPHM allows the client to report faults to a specified PHM instance.,
 * D,
 * DR-iAOS-RTF-RTFPHM-00047
 * }
 */
constexpr ara::core::ErrorDomain const &GetPhmErrorDomain() noexcept
{
    return g_PhmErrorDomain;
}
/**
 * @ingroup DetectorProxy
 * @brief Make the error code.
 * @param[in] code Phm error code.
 * @param[in] data Support data type.
 * @return ara::core::ErrorCode
 * @req{AR-iAOS-RTF-RTFPHM-00047,
 * RTFPHM allows the client to report faults to a specified PHM instance.,
 * D,
 * DR-iAOS-RTF-RTFPHM-00047
 * }
 */
constexpr ara::core::ErrorCode MakeErrorCode(ara::phm::PhmErrc code,
                                             ara::core::ErrorDomain::SupportDataType data = 0) noexcept
{
    return ara::core::ErrorCode(static_cast<ara::core::ErrorDomain::CodeType>(code), GetPhmErrorDomain(), data);
}
} // namespace phm
} // namespace ara

#endif // VRTF_PHM_ERROR_DOMAIN_H

