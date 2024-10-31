/*!
 * @file diag_server_event_sub.h
 * This header file contains the declaration of the sub functions.
 *
 */

#pragma once

#include "cm/include/skeleton.h"
#include "cm/include/proxy.h"
#include "idl/generated/phmPubSubTypes.h"
#include "idl/generated/diagPubSubTypes.h"
#include <memory>
#include <functional>

namespace hozon {
namespace netaos {
namespace diag {

class DiagServerEventPub
{
public:
    DiagServerEventPub();
    virtual ~DiagServerEventPub();

    void init();
    void deInit();
    void sendFaultEvent(uint32_t faultKey, uint8_t status);
    void notifyDtcControlSetting(const uint8_t dtcControlSetting);
    void notifyHmi();

private:
    std::shared_ptr<fault_eventPubSubType> m_spFaultEventPub;
    std::shared_ptr<hozon::netaos::cm::Skeleton> m_spSkeleton;

    std::shared_ptr<dtcControlSettingSwPubSubType> m_spDtcControlSettingSwPub;
    std::shared_ptr<hozon::netaos::cm::Skeleton> m_spDtcControlSettingSwSkeleton;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
