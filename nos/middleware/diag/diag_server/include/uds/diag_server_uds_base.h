
#ifndef DIAG_SERVER_UDS_BASE_H
#define DIAG_SERVER_UDS_BASE_H

#include "diag/diag_server/include/common/diag_server_def.h"
#include "diag/diag_server/include/common/diag_server_logger.h"
#include "diag/diag_server/include/common/diag_server_config.h"
#include "diag_server_transport_cm.h"

namespace hozon {
namespace netaos {
namespace diag {

using namespace hozon::netaos::diag::cm_transport;

class DiagServerUdsBase {

public:
    DiagServerUdsBase();
    virtual ~DiagServerUdsBase();

    virtual void AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage);
    virtual void PositiveResponse(const DiagServerUdsMessage& udsMessage);
    virtual void NegativeResponse(const DiagServerUdsMessage& udsMessage);

private:
    DiagServerUdsBase(const DiagServerUdsBase &);
    DiagServerUdsBase & operator = (const DiagServerUdsBase &);
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_UDS_SID_BASE_H
