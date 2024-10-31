
#ifndef DOSOMEIP_MANAGER_H
#define DOSOMEIP_MANAGER_H

#include <stdint.h>
#include <functional>
#include <mutex>
#include "diag/dosomeip/common/dosomeip_def.h"
#include "diag/dosomeip/someip/src/DoSomeIPSkeleton.h"

namespace hozon {
namespace netaos {
namespace diag {

class DoSomeIPManager {
public:
    DoSomeIPManager();
    virtual ~DoSomeIPManager(){};

    DOSOMEIP_RESULT Init(std::function<void(const DoSomeIPReqUdsMessage&)> uds_request_callback, std::function<void(const DOSOMEIP_REGISTER_STATUS&)> someip_register_callback);
    void DeInit();

    DOSOMEIP_RESULT DispatchUDSReply(const DoSomeIPRespUdsMessage& udsMsg);

private:
    DoSomeIPManager(const DoSomeIPManager&);
    DoSomeIPManager& operator=(const DoSomeIPManager&);

private:
    std::unique_ptr<DoSomeIPSkeleton> skeleton_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DOSOMEIP_MANAGER_H