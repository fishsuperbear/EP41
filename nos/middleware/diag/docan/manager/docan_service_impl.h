/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanServiceImpl Header
 */

#ifndef DOCAN_SERVICE_IMPL_H_
#define DOCAN_SERVICE_IMPL_H_
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <string>
#include <vector>
#include <memory>

namespace hozon {
namespace netaos {
namespace diag {

    class DocanListener;
    /**
     * @brief Class of DocanServiceImpl
     *
     * This class is a implement of DocanServiceImpl.
     */
    class DocanServiceImpl
    {
    public:
        DocanServiceImpl();
        virtual ~DocanServiceImpl();

        int32_t Init();
        int32_t Start();
        int32_t Stop();
        int32_t Deinit();

        int32_t registerListener(const std::string& who, const std::shared_ptr<DocanListener>& listener);
        int32_t unregisterListener(const std::string& who);

        int32_t UdsRequest(const std::string& who, uint16_t reqSa, uint16_t reqTa, const std::vector<uint8_t>& uds);

    private:
        DocanServiceImpl(const DocanServiceImpl&);
        DocanServiceImpl& operator=(const DocanServiceImpl&);

    };


} // end of diag
} // end of netaos
} // end of hozon
#endif  // DOCAN_SERVICE_IMPL_H_
/* EOF */
