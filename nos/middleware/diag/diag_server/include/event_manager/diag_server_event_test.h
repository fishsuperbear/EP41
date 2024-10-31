/*!
 * @file diag_server_event_test.h
 * This header file contains the declaration of the sub functions.
 *
 */

#pragma once

#include "cm/include/skeleton.h"
#include "cm/include/proxy.h"
#include "idl/generated/diagPubSubTypes.h"
#include <memory>
#include <functional>

namespace hozon {
namespace netaos {
namespace diag {

class DiagServerEventTest
{
public:
    DiagServerEventTest();
    virtual ~DiagServerEventTest();

    void init();
    void deInit();
    void recvTestCallback();

private:
    std::shared_ptr<testDiagEventPubSubType> m_spTestSub;
    std::shared_ptr<hozon::netaos::cm::Proxy> m_spTestProxy;
    std::shared_ptr<testDiagEvent> m_testDiagEventData;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon