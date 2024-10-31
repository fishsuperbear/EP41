/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanStateManager Header
 */

#ifndef DOCAN_STATE_MANAGER_H_
#define DOCAN_STATE_MANAGER_H_
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <memory>
#include <mutex>
#include <list>
#include <map>

namespace hozon {
namespace netaos {
namespace diag {

    /**
     * @brief Class of DocanStateManager
     *
     * this class used to save the Docan device info.
     */
    class DocanStateManager
    {
    public:
        static DocanStateManager* instance();
        static void               destroy();

        int32_t         Init(void);
        int32_t         Start(void);
        int32_t         Stop(void);
        int32_t         Deinit(void);

    private:
        DocanStateManager();
        virtual ~DocanStateManager();

        DocanStateManager(const DocanStateManager&);
        DocanStateManager& operator=(const DocanStateManager&);

    private:
        mutable std::mutex        m_sync;

        static DocanStateManager *s_instance;
        static std::mutex s_instance_mutex;

    };


} // end of diag
} // end of netaos
} // end of hozon
#endif  // DOCAN_STATE_MANAGER_H_
/* EOF */