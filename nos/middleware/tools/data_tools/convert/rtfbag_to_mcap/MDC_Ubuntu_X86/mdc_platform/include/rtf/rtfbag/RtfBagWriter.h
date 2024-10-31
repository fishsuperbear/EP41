/* Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: rtf writer bag file
 * Author:
 * Create: 2022-12-06
 * Notes: NA
 * History: 2022-12-06
 */
#ifndef RTF_BAG_WRITER_H
#define RTF_BAG_WRITER_H
#include "rtf/cm/config/entity_index_info.h"
#include "rtf/maintaind/impl_type_appregisterinfo.h"
#include "rtf/internal/RtfBagFile.h"
#include "ara/core/variant.h"
namespace rtf {
namespace rtfbag {
using DDSEventIndexInfo = rtf::cm::config::DDSEventIndexInfo;
using SOMEIPEventIndexInfo = rtf::cm::config::SOMEIPEventIndexInfo;

enum class ErrorCode : uint8_t {
    SUCCESS = 0x00,          // 表示接口调用成功
    ERROR,                   // 表示接口调用存在错误
    INVALID_CONFIG_PATH,     // 表示对应的路径无法解析配置文件
    NO_EXISTED_ENTITY,       // 找不到对应需要解析的event
    INVALID_BAG_NAME,        // 错误的bag文件名称
    INVALID_BAG_VERSION,     // 错误的bag文件版本号
    INVALID_BAG_PATH,        // 错误的bag文件生成路径
    WARN_BAG_NOT_CREATED,    // 文件未成功创建
    WARN_WRITE_SAME_FILE,    // 存在同名active文件
    WARN_COMPRESS_INIT_FAIL, // 压缩模块初始化失败
    WARN_CREATE_FILE_FAIL,   // 创建文件失败
    WARN_WRITE_EMPTY_MSG,    // 写入了一个空数据
    WARN_EVENT_ENTITY_ERROR, // 获取event配置信息失败
    ERROR_COMPRESS_FAIL,     // 文件数据压缩失败
    ERROR_WRITE_MSG_FAIL,    // 写入数据失败
    ERROR_CLOSE_TO_FILE,     // 关闭文件失败
    ERROR_RENAME_TO_BAG      // 将active文件重命名为bag文件失败
};

class EventMsgInfo {
public:
    EventMsgInfo() = default;
    virtual ~EventMsgInfo() = default;
    EventMsgInfo(const EventMsgInfo&) = delete;
    EventMsgInfo(EventMsgInfo &&) = delete;
    EventMsgInfo &operator=(EventMsgInfo &&) & = delete;
    EventMsgInfo& operator=(EventMsgInfo const &) & = delete;

    virtual ErrorCode Init() = 0;
    virtual void SetBuffer(std::vector<std::uint8_t> && buff) = 0;
    virtual const std::vector<std::uint8_t>& GetBuffer() const = 0;
    virtual bool GetEventInfoStatus() const = 0;
    virtual const rtf::maintaind::EventRegisterInfo& GetEventInfo() const = 0;
    virtual const std::string& GetEventDataTypeRef() const = 0;
};

class RtfBagWriter {
public:
    explicit RtfBagWriter(const std::string& bagName, const std::string& bagVersion = "2.3", 
                const FileCompressionType& type = FileCompressionType::NONE, const std::string& bagPath = "");
    ~RtfBagWriter();
    ErrorCode Init();
    ErrorCode Write(const std::shared_ptr<EventMsgInfo>& eventMsgInfo, const uint64_t timeStamp);
    ErrorCode Stop();
private:
    ErrorCode CheckInputParam();
    void BagFileSync();
    std::string bagName_{};
    std::string targetFileName_{};
    std::string writeFileName_{};
    std::string pureFileName_{};
    std::string bagVersion_{"1.0"};
    std::string bagPath_{};
    FileCompressionType type_{FileCompressionType::NONE};
    std::shared_ptr<rtf::rtfbag::RtfBagFile> bag_ {nullptr};
    bool isBagCreated_ {false};
    std::mutex lock_;
};

class EventMsgFactory {
public:
    static std::shared_ptr<EventMsgInfo> CreateEventMsgFactory(
        const std::string& etcPath, const DDSEventIndexInfo& index, const std::string& msgPath) noexcept;
    static std::shared_ptr<EventMsgInfo> CreateEventMsgFactory(
        const std::string& etcPath, const SOMEIPEventIndexInfo & index, const std::string& msgPath) noexcept;
};
}  // namespace rtfbag
}  // namespace rtf
#endif // RTF_PLAYER_H
