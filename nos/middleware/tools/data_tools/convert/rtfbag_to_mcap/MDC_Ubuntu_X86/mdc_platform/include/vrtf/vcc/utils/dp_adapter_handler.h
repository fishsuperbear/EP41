/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: Implement dp adapter operations
 * Create: 2020-06-10
 */
#ifndef VRTF_DPADAPTERHANDLER_H
#define VRTF_DPADAPTERHANDLER_H
#include <memory>
#include <mutex>
#include "vrtf/driver/dds/mbuf.h"
#include "ara/hwcommon/log/log.h"

namespace vrtf {
namespace vcc {
namespace utils {
constexpr std::size_t MBUF_PRIVATE_DATA_SIZE {256U};
constexpr std::size_t MAX_E2E_HEADER_SIZE {20U};
/* Now use 4 types for record data size */
constexpr std::size_t CM_RESERVE_SIZE {24U};
using MbufFreeFunc = int32_t (*)(Mbuf *mbuf);
class DpAdapterHandler {
public:
    DpAdapterHandler();
    ~DpAdapterHandler();
    static std::shared_ptr<DpAdapterHandler> GetInstance() noexcept;
    int32_t MbufGetPrivInfo(Mbuf *mbuf,  void **priv, uint32_t *size) const;
    int32_t MbufGetDataPtr(Mbuf *mbuf, void **buf, uint64_t *size) const;
    MbufFreeFunc GetMbufFreeFunc() const noexcept;
    int32_t MbufFree(Mbuf *mbuf) const noexcept;
    std::size_t GetAvailabeLenth() const;
    int32_t BuffCreatePool(DP_MemPoolAttr *attr, PoolHandle *pHandle) const;
    int32_t BuffDeletePool(PoolHandle pHandle) const;
    int32_t MbufAllocByPool(PoolHandle pHandle, Mbuf **mbuf) const;
    int32_t MbufAlloc(uint64_t size, Mbuf **mbuf) const;
    int32_t MbufSetDataLen(Mbuf *mbuf, uint64_t len) const;
    int32_t MbufGetDataLen(Mbuf *mbuf, uint64_t *len) const;
private:
    const DPAdapterOps *dpAdapterOps_ = nullptr;
    std::shared_ptr<ara::godel::common::log::Log> logInstance_;
    std::size_t availableSize_ = 0U;
    DPAdapterOps optDefault_;
};
}
}
}
#endif // VRTF_DPADAPTERHANDLER_H
