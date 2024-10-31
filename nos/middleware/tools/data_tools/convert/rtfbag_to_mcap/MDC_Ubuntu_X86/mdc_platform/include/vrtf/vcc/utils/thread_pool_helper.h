/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: The declaration of ThreadPoolHelper
 * Create: 2020-08-17
 */
#ifndef VRTF_VCC_UTILS_THREAD_POOL_HELPER_H
#define VRTF_VCC_UTILS_THREAD_POOL_HELPER_H

#include <mutex>
#include "vrtf/vcc/utils/vcc_thread_group.h"
#include "vrtf/vcc/api/param_struct_typse.h"

namespace vrtf {
namespace vcc {
namespace utils {
class ThreadPool;
class ThreadPoolHelper {
public:
    /**
     * @brief Create a thread pool which index is inputted threadgroup
     *
     * @param[in] group           The index of new thread pool
     * @param[in] threadNumber    The thread number of the new thread pool
     * @param[in] queueSize       The queue size of the new thread pool
     */
    static bool AddThreadPool(const VccThreadGroup& group) noexcept;

    /**
     * @brief Get the corresponding thread pool of the thread group
     *
     * @param[in] group   the index of thread pool
     * @return std::shared_ptr<VccThreadPool> query result
     */
    static std::shared_ptr<ThreadPool> QueryThreadPool(const VccThreadGroup& group) noexcept;

    /**
     * @brief Delete the Thread pool from helper(manager)
     *
     * @param[in] group  The index of the deleting thread pool
     */
    static void DeleteThreadPool(const VccThreadGroup& group) noexcept;
private:
    static std::mutex threadPoolMapMutex_;
};
}
}
}
#endif
