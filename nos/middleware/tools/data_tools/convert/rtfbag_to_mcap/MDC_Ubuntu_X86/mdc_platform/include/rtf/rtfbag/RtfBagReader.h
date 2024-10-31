/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: Get the specified bag and event message information
 * Create: 2021-09-03
 */
#ifndef RTF_BAG_READER_H
#define RTF_BAG_READER_H

#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "rtf/internal/RtfBagFile.h"
#include "rtf/internal/RtfLog.h"
#include "rtf/internal/RtfView.h"

namespace rtf {
namespace rtfbag {
class RtfBagReader {
public:
    enum class EventProtocol : uint8_t {
        DDS = 0x00,
        SOMEIP = 0x01,
        UNKNOWN = 0xFF
    };
    enum class CompressionType : uint32_t {
        NONE = 0x00,
        GZIP = 0x01,
        ZLIB = 0x02
    };
    enum class ErrorCode : uint8_t {
        SUCCESS = 0x00, // 表示接口调用成功
        INIT_FAIL = 0x01, // 表示初始化失败
        HAS_NEXT_FRAME = 0x02, // 表示数据读取成功，并且读取列表中存在下一帧数据
        LAST_FRAME = 0x03, // 表示数据读取成功，并且当前读取的数据帧为最后一帧数据
        WRONG_READ = 0x04, // 表示读取到最后一帧数据后调用接口进行获取或者读取的指定长度超出消息的最大长度
        WRONG_INPUT = 0x05, // 表示输入的Event或者指定的数据帧不存在
        DECOMPRESS_ERROR = 0x06, // 表示解压存在失败
        UNKNOWN_ERROR = 0xFF // 其他未定义错误
    };
    class RtfEventInfo {
    public:
        void SetDataType(const std::string &dataType) { dataType_ = dataType; }
        std::string GetDataType() const { return dataType_; }
        void SetProtocol(const EventProtocol protocol) { protocol_ = protocol; }
        EventProtocol GetProtocol() const { return protocol_; }
        void SetMsgCount(const uint32_t msgCount) { msgCount_ = msgCount; }
        uint32_t GetMsgCount() const { return msgCount_; }
    private:
        std::string dataType_;
        EventProtocol protocol_ = EventProtocol::UNKNOWN;
        uint32_t msgCount_ = 0;
    };
    class RtfBagInfo {
    public:
        void SetBagVersion(const std::string &bagVersion) { bagVersion_ = bagVersion; }
        std::string GetBagVersion() const { return bagVersion_; }
        void AddToEventInfoList(const std::string &eventName,
            const RtfEventInfo &eventInfo) { eventInfoList_[eventName] = eventInfo; }
        const std::map<std::string, RtfEventInfo>& GetEventInfoList() const { return eventInfoList_; }
        void SetMsgCount(const uint32_t msgCount) { msgCount_ = msgCount; }
        uint32_t GetMsgCount() const { return msgCount_; }
        void SetBeginTime(const uint64_t beginTime) { beginTime_ = beginTime; }
        uint64_t GetBeginTime() const { return beginTime_; }
        void SetEndTime(const uint64_t endTime) { endTime_ = endTime; }
        uint64_t GetEndTime() const { return endTime_; }
        void SetBagSize(const uint64_t bagSize) { bagSize_ = bagSize; }
        uint64_t GetBagSize() const { return bagSize_; }
        void SetCompression(const CompressionType compression) { compression_ = compression; }
        CompressionType GetCompression() const { return compression_; }
    private:
        std::string bagVersion_ = "1.0";
        std::map<std::string, RtfEventInfo> eventInfoList_;
        uint32_t msgCount_ = 0;
        uint64_t beginTime_ = 0;
        uint64_t endTime_ = 0;
        uint64_t bagSize_ = 0;
        CompressionType compression_ = CompressionType::NONE;
    };
    class EventMsg {
    public:
        void SetBuffer(std::vector<std::uint8_t> && buff) { buffer_ = std::move(buff); }
        const std::vector<std::uint8_t>& GetBuffer() const { return buffer_; }
        size_t GetBufferSize() const { return buffer_.size(); }
        void SetTimeStamp(const uint64_t timeStamp) { timeStamp_ = timeStamp; }
        uint64_t GetTimeStamp() const { return timeStamp_; }
        void SetEventName(const std::string &eventName) { eventName_ = eventName; }
        std::string GetEventName() const { return eventName_; }
    private:
        std::vector<std::uint8_t> buffer_;
        uint64_t timeStamp_ = 0;
        std::string eventName_;
    };

    explicit RtfBagReader(const std::string& bagPath);
    ~RtfBagReader() = default;
    RtfBagReader(const RtfBagReader& other);
    RtfBagReader& operator=(const RtfBagReader& other);
    ErrorCode Init();
    RtfBagInfo GetBagInfo() const;
    // 按文件结构顺序读取一帧event的数据,调用一次读取一帧
    ErrorCode ReadEventMsg(EventMsg &eventMsg, const uint32_t len = 0);
    // 读取指定event name的第N帧数据
    ErrorCode ReadEventMsg(const std::string &eventName, const uint32_t seq, EventMsg &eventMsg,
        const uint32_t len = 0);
    // 按时间序获取指定event name的一帧数据，调用一次读取一帧
    ErrorCode ReadEventMsg(const std::string &eventName, EventMsg &eventMsg, const uint32_t len = 0);
private:
    RtfBagReader& operator=(RtfBagReader && other) = delete;
    RtfBagReader(RtfBagReader && other) = delete;
    class EventMsgView {
    public:
        EventMsgView(RtfBagFile& bagfile, uint64_t const& startTime, uint64_t const& endTime,
            const std::string &eventname = "");
        ~EventMsgView() = default;
        EventMsgView(const EventMsgView& other);
        EventMsgView& operator=(const EventMsgView& other);
        EventMsgView& operator++();
        RtfMsgEntity GetPosValue(const uint32_t pos) { return eventMsgView_.Begin().GetPosValue(pos); }
        bool IsEnd() const { return eventMsgViewInter_.IsEnd(); }
        RtfMsgEntity Value() { return eventMsgViewInter_.Value(); }
    private:
        EventMsgView& operator=(EventMsgView && other) = delete;
        EventMsgView(EventMsgView && other) = delete;
        RtfView eventMsgView_;
        RtfView::Iterator eventMsgViewInter_;
    };
    void CopyCtorToMakeEventViews(const RtfBagReader& other);
    void GetBagInfoAndMakeViews(const BagFileHeadInfo &headInfo);
    void MakeEventViews();
    ErrorCode GetNextEventMsg(EventMsg &eventMsg, EventMsgView &eventMsgView, const uint32_t len) const;
    ErrorCode ChangeToEventMsg(EventMsg &eventMsg, const RtfMsgEntity& msgEntity, const uint32_t len) const;
    std::string bagPath_;
    std::shared_ptr<RtfBagFile> rtfbagFile_ = nullptr;
    std::shared_ptr<EventMsgView> allBagView_ = nullptr;
    std::map<std::string, std::shared_ptr<EventMsgView>> eventMsgViewList_;
    std::once_flag initFlag_;
    ErrorCode initResult_ = ErrorCode::UNKNOWN_ERROR;
    std::shared_ptr<RtfLog::Log> logger_ = nullptr;
    RtfBagInfo bagInfo_;
    // behavior mutex, can not support publish use multi thread
    std::mutex lock_;
};
}
}
#endif
