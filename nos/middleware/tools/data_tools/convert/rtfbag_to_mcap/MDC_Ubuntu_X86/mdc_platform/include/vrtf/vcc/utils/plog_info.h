/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: print plog to cal delay info
 * Create: 2020-12-02
 */
#ifndef VRTF_VCC_PLOG_INFO_H
#define VRTF_VCC_PLOG_INFO_H
#include <memory>
#include <cstdint>
namespace rbs {
namespace plog {
class ProfileLogWriter;
}
}
namespace vrtf {
namespace vcc {
namespace utils {
struct TimeNodeStage;
using MoudleID = uint8_t;
using ModuleID = std::uint8_t;
ModuleID constexpr CM_SEND = 1U;
ModuleID constexpr CM_RECV = 2U;
ModuleID constexpr SOMEIP_RECV = 4U;
ModuleID constexpr DDS_RECV = 6U;

// PlogServerTimeStampNode::SEND_OVER + 1 is the size of TimeNodeContainer for server
// Keep its value continuous to the end of SEND_OVER
enum class PlogServerTimeStampNode: std::uint8_t {
    USER_SEND_EVENT = 0x00,
    PULL_THREADPOOL,
    TRIGGER_TASK,
    SERIALIZE_DATA,
    SEND_OVER
};

// PlogClientTimeStampNode::NOTIFY_USER + 1 is the size of TimeNodeContainer for client
// Keep its value continuous to the end of NOTIFY_USER
enum class PlogClientTimeStampNode: std::uint8_t {
    RECVIVE_NOTIFY = 0x00,
    USER_TAKE,
    TAKE_FROM_DRIVER,
    DESERIALIZE_DATA,
    NOTIFY_USER,
};

enum class PlogDriverType: std::uint8_t {
    COMMON = 0x00,
    DDS,
    SOMEIP
};

class PlogInfo {
public:
    /**
     * @brief PlogInfo construct
     * @param[in] id this plog module id(include CM_SEND, CM_RECV).
     */
    PlogInfo() = default;

    /**
     * @brief PlogInfo deconstruct
     */
    virtual ~PlogInfo() = default;
    /**
     * @brief Create PlogInfo shared pointer
     * @param[in] this plog msg module id
     * @return std::shared_ptr<PlogInfo> make one plogInfo
     */
    static std::shared_ptr<PlogInfo> CreatePlogInfo(const MoudleID& id);

    /**
     * @brief Get PlogInfoInstance to maintain life cycle
     * @param[in] this plog msg module id
     * @return std::shared_ptr<ProfileLogWriter>
     */
    static std::shared_ptr<rbs::plog::ProfileLogWriter> GetPlogInstance(const MoudleID& id);

    /**
     * @brief initialize plog module
     * @param[in] this plog msg module id
     */
    static void InitPlog(const MoudleID& id);

    static bool GetSendFlag();

    static bool GetRecvFlag();
    /**
     * @brief get now time stamp
     * @return timespec now time stamp
     */
    static std::uint64_t GetPlogTimeStamp();

    /**
     * @brief use for server, store this stage node corresponding timespec
     * @param[in] node which node to record time.
     */
    virtual void WriteTimeStamp(
        const PlogServerTimeStampNode& node, const PlogDriverType& type) = 0;

    /**
     * @brief use for client, store this stage node corresponding timespec
     * @param[in] node which node to record time.
     * @param[in] time the time when this node trigger.
     */
    virtual void WriteTimeStamp(const PlogClientTimeStampNode& node, const std::uint64_t time) = 0;

    /**
     * @brief Send store time and send to plog module
     */
    virtual void SendPlogStamp(const PlogDriverType& type) = 0;

    /**
     * @brief use for client, binding this plog with driver uid and driver moduleId
     * @param[in] uid corresponding driver uid
     * @param[in] moudleId the module which send msg to client
     */
    virtual void RelatedModule(const std::uint64_t& uid, const ModuleID& moudleId) = 0;

    /**
     * @brief Get this plog uid
     * @return std::uint64_t this plog msg's uid
     */
    virtual std::uint64_t GetMsgUid(const PlogDriverType& type) = 0;
};
}
}
}
#endif
