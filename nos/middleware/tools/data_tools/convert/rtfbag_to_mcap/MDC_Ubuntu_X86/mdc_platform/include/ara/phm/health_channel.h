/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: HealthChannel definition
 * Create: 2020-3-27
 */
#ifndef ARA_PHM_HEALTH_CHANNEL_H
#define ARA_PHM_HEALTH_CHANNEL_H

#include <string>
#include <rtf/com/rtf_com.h>
#include <json_parser/document.h>
#include <json_parser/global.h>

#include "ara/core/instance_specifier.h"
#include "ara/core/result.h"

namespace ara {
namespace phm {
using HealthStatus = uint32_t;
using CommunicationMode = std::set<rtf::com::TransportMode>;
/**
 * @defgroup DetectorProxy DetectorProxy
 * @brief Container for all HealthChannel objects.
 * @ingroup DetectorProxy
 */
/* AXIVION Next Line AutosarC++19_03-A12.0.1, AutosarC++19_03-A0.1.6 : Standard external interface,
 * can't add other function, the unused type is offered to the user [SWS_PHM_00457] */
class HealthChannel final {
public:
    /**
     * @ingroup DetectorProxy
     * @brief Constructor of HealthChannel.[SWS_PHM_01122][SWS_PHM_01139][SWS_PHM_00457]
     * @par Description
     * The Platform Health Management provides constructor for class HealthChannel
     * indicating the instance specifier of the HealthChannel.
     * @param[in] instance Instance id input.
     * @return HealthChannel
     * @req{AR-iAOS-RTF-RTFPHM-00026,
     * The RTFPHM supports the reporting of a single client to the health channel.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00026
     * }
     */
    explicit HealthChannel(ara::core::InstanceSpecifier const &instance);
    /**
     * @ingroup DetectorProxy
     * @brief Constructor of HealthChannel.
     * @param[in] instance Instance id input.
     * @param[in] processName The name of the process to report.
     * @param[in] communicationMode The communication mode that HealthChannel will use.
     * @return HealthChannel
     * @req{AR-iAOS-RTF-RTFPHM-00026,
     * The RTFPHM supports the reporting of a single client to the health channel.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00026
     * }
     */
    HealthChannel(const core::InstanceSpecifier &instance, std::string const &processName,
                  CommunicationMode const &communicationMode);
    /**
     * @ingroup DetectorProxy
     * @brief Constructor of HealthChannel.
     * @param[in] instance Instance id input.
     * @param[in] communicationMode The communication mode that HealthChannel will use.
     * @return HealthChannel
     * @req{AR-iAOS-RTF-RTFPHM-00026,
     * The RTFPHM supports the reporting of a single client to the health channel.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00026
     * }
     */
    HealthChannel(const core::InstanceSpecifier &instance, CommunicationMode const &communicationMode);
    /**
     * @ingroup DetectorProxy
     * @brief Destructor of HealthChannel.
     * @req{AR-iAOS-RTF-RTFPHM-00026,
     * The RTFPHM supports the reporting of a single client to the health channel.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00026
     * }
     */
    ~HealthChannel() = default;
    /**
     * @ingroup DetectorProxy
     * @brief Report the health status to phm server.[SWS_PHM_01128]
     * @par Description
     * The Platform Health Management provides a method ReportHealthStatus, provided by HealthChannel.
     * @param[in] healthStatusId The health status id to report.
     * @return ara::core::Result<void>
     * @req{AR-iAOS-RTF-RTFPHM-00026,
     * The RTFPHM supports the reporting of a single client to the health channel.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00026
     * }
     */
    ara::core::Result<void> ReportHealthStatus(HealthStatus healthStatusId);
    /**
     * @ingroup DetectorProxy
     * @brief Enable the E2E.
     * @param[in] dataId The data id.
     * @return void
     * @req{AR-iAOS-RTF-RTFPHM-00026,
     * The RTFPHM supports the reporting of a single client to the health channel.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00026
     * }
     */
    static void EnableE2E(std::uint32_t dataId);
    /**
     * @ingroup DetectorProxy
     * @brief Set the CRC validation level.
     * @param[in] crcOption The CRC validation level.
     * @return void
     * @req{AR-iAOS-RTF-RTFPHM-00026,
     * The RTFPHM supports the reporting of a single client to the health channel.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00026
     * }
     */
    static void SetCrcOption(ara::godel::common::jsonParser::CRCVerificationType const crcOption);
private:
    /**
     * @ingroup DetectorProxy
     * @brief Add hc client instance.
     * @param[in] communicationMode.
     * @return void
     * @req{AR-iAOS-RTF-RTFPHM-00026,
     * The RTFPHM supports the reporting of a single client to the health channel.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00026
     * }
     */
    void AddInstance(CommunicationMode const &communicationMode = {});
    ara::core::String instance_ {};
    std::string processName_ {};
    rtf::com::InstanceId instanceId_ {};
};
} // namespace phm
} // namespace ara
#endif

