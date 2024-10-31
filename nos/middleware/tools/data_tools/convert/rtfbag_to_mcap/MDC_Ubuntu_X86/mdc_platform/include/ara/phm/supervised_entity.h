/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: SupervisedEntity definition
 * Create: 2020-3-27
 */
#ifndef ARA_PHM_SUPERVISED_ENTITY_H
#define ARA_PHM_SUPERVISED_ENTITY_H

#include <string>
#include <rtf/com/rtf_com.h>
#include <json_parser/document.h>
#include <json_parser/global.h>

#include "ara/core/instance_specifier.h"
#include "ara/core/result.h"

namespace ara {
namespace phm {
using Checkpoint = uint32_t;

enum class LocalSupervisionStatus : uint32_t {
    DEINIT      = 0U,
    K_DEACTIVATED = 1U,
    K_OK          = 2U,
    K_FAILED      = 3U,
    K_EXPIRED     = 4U
};

enum class GlobalSupervisionStatus : uint32_t {
    DEINIT      = 0U,
    K_DEACTIVATED = 1U,
    K_OK          = 2U,
    K_FAILED      = 3U,
    K_EXPIRED     = 4U,
    K_STOPPED     = 5U
};
using CommunicationMode = std::set<rtf::com::TransportMode>;
/**
 * @defgroup DetectorProxy DetectorProxy
 * @brief Container for all SupervisedEntity objects.
 * @ingroup DetectorProxy
 */
/* AXIVION Next Line AutosarC++19_03-A12.0.1, AutosarC++19_03-A0.1.6 : Standard external interface,
 * can't add other function, the unused type is offered to the user [SWS_PHM_01123] */
class SupervisedEntity final {
public:
    /**
     * @ingroup DetectorProxy
     * @brief Constructor of SupervisedEntity.[SWS_PHM_01138][SWS_PHM_01136][SWS_PHM_01137][SWS_PHM_01132]
     *        [SWS_PHM_01123]
     * @par Description
     * The Platform Health Management provides constructor for class SupervisedEntity
     * indicating the instance specifier of the SupervisedEntity.
     * @param[in] instance Instance id input.
     * @return SupervisedEntity
     * @req{SAR-iAOS3-RTF-RTFPHM-00005,
     * Checkpoint interfaces reported by apps must correctly report status to the PHM.,
     * D,
     * SDR-iAOS3-RTF-RTFPHM-00005
     * }
     */
    explicit SupervisedEntity(ara::core::InstanceSpecifier const &instance);
    /**
     * @ingroup DetectorProxy
     * @brief Constructor of SupervisedEntity.
     * @param[in] instance Instance id input.
     * @param[in] processName The name of the process to report.
     * @param[in] communicationMode The communication mode that phm client will use.
     * @return SupervisedEntity
     * @req{SAR-iAOS3-RTF-RTFPHM-00005,
     * Checkpoint interfaces reported by apps must correctly report status to the PHM.,
     * D,
     * SDR-iAOS3-RTF-RTFPHM-00005
     * }
     */
    SupervisedEntity(const core::InstanceSpecifier &instance, std::string const &processName,
                     CommunicationMode const &communicationMode);
    /**
     * @ingroup DetectorProxy
     * @brief Constructor of SupervisedEntity.
     * @param[in] instance Instance id input.
     * @param[in] communicationMode The communication mode that phm client will use.
     * @return SupervisedEntity
     * @req{SAR-iAOS3-RTF-RTFPHM-00005,
     * Checkpoint interfaces reported by apps must correctly report status to the PHM.,
     * D,
     * SDR-iAOS3-RTF-RTFPHM-00005
     * }
     */
    SupervisedEntity(const core::InstanceSpecifier &instance, CommunicationMode const &communicationMode);
    /**
     * @ingroup DetectorProxy
     * @brief Destructor of SupervisedEntity.
     * @req{SAR-iAOS3-RTF-RTFPHM-00005,
     * Checkpoint interfaces reported by apps must correctly report status to the PHM.,
     * D,
     * SDR-iAOS3-RTF-RTFPHM-00005
     * }
     */
    ~SupervisedEntity() = default;
    /**
     * @ingroup DetectorProxy
     * @brief Report checkpoint to phm server.[SWS_PHM_01127]
     * @par Description
     * The Platform Health Management provides a method ReportCheckpoint, provided by SupervisedEntity.
     * @param[in] checkpointId The checkpoint id to report.
     * @return ara::core::Result<void>
     * @req{SAR-iAOS3-RTF-RTFPHM-00005,
     * Checkpoint interfaces reported by apps must correctly report status to the PHM.,
     * D,
     * SDR-iAOS3-RTF-RTFPHM-00005
     * }
     */
    ara::core::Result<void> ReportCheckpoint(Checkpoint checkpointId);
    /**
     * @ingroup DetectorProxy
     * @brief Get the local supervision status.[SWS_PHM_01134]
     * @par Description
     * The Platform Health Management provides a method GetLocalSupervisionStatus, provided by SupervisedEntity.
     * This method returns the current Local Supervision Status of this SupervisedEntity.
     * @return ara::core::Result<LocalSupervisionStatus>
     * @req{SAR-iAOS3-RTF-RTFPHM-00005,
     * Checkpoint interfaces reported by apps must correctly report status to the PHM.,
     * D,
     * SDR-iAOS3-RTF-RTFPHM-00005
     * }
     */
    ara::core::Result<LocalSupervisionStatus> GetLocalSupervisionStatus() const;
    /**
     * @ingroup DetectorProxy
     * @brief Get the global supervision status.[SWS_PHM_01135]
     * @par Description
     * The Platform Health Management provides a method GetGlobalSupervisionStatus, provided by SupervisedEntity.
     * This method returns the current Global Supervision Status corresponding to this SupervisedEntity
     * @return ara::core::Result<GlobalSupervisionStatus>
     * @req{SAR-iAOS3-RTF-RTFPHM-00005,
     * Checkpoint interfaces reported by apps must correctly report status to the PHM.,
     * D,
     * SDR-iAOS3-RTF-RTFPHM-00005
     * }
     */
    ara::core::Result<GlobalSupervisionStatus> GetGlobalSupervisionStatus() const;
    /**
     * @ingroup DetectorProxy
     * @brief Enable E2E.
     * @param[in] dataId The data id.
     * @return void
     * @req{SAR-iAOS3-RTF-RTFPHM-00005,
     * Checkpoint interfaces reported by apps must correctly report status to the PHM.,
     * D,
     * SDR-iAOS3-RTF-RTFPHM-00005
     * }
     */
    static void EnableE2E(std::uint32_t dataId);
    /**
     * @ingroup DetectorProxy
     * @brief Set the CRC validation level.
     * @param[in] crcOption The CRC validation level.
     * @return void
     * @req{SAR-iAOS3-RTF-RTFPHM-00005,
     * Checkpoint interfaces reported by apps must correctly report status to the PHM.,
     * D,
     * SDR-iAOS3-RTF-RTFPHM-00005
     * }
     */
    static void SetCrcOption(ara::godel::common::jsonParser::CRCVerificationType const crcOption);
private:
    ara::core::String instance_ {};
    std::string processName_ {};
    rtf::com::InstanceId instanceId_ {};
};
} // namespace phm
} // namespace ara
#endif

