/*!
 * @file diag_server_event_sub.h
 * This header file contains the declaration of the sub functions.
 *
 */

#pragma once

#include "cm/include/proxy.h"
#include "idl/generated/diagPubSubTypes.h"
#include <memory>
#include <functional>

namespace hozon {
namespace netaos {
namespace diag {

class DiagServerEventSub
{
public:
    DiagServerEventSub();
    virtual ~DiagServerEventSub();

    void init();
    void deInit();
    void registCallback();
    void recvCallback();

private:

    std::shared_ptr<reportDemEventPubSubType> m_spPubsubtype;
    std::shared_ptr<hozon::netaos::cm::Proxy> m_spProxy;
    std::shared_ptr<reportDemEvent> m_spDemData;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon