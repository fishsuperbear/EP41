/* Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: rtf fix
 * Create: 2021-08-14
 * Notes: NA
 */
#ifndef RTF_FIX_H
#define RTF_FIX_H

#include <atomic>
#ifndef RTFBAG_FIX_CFLOAT
#define RTFBAG_FIX_CFLOAT
#include <cfloat>
#endif
#include <chrono>
#include <functional>
#include <memory>

#include "ara/core/map.h"
#include "ara/core/string.h"
#include "ara/core/vector.h"
#include "rtf/internal/RtfBagFile.h"
#include "rtf/internal/RtfMsgEntity.h"
#include "rtf/internal/RtfView.h"

namespace rtf {
namespace rtfbag {
const uint32_t MIN_SPACE = 1073741824U;
enum class FixPrintType : uint8_t {
    OPEN_FILE  = 0,     // 打开文件
    OPEN_FAIL,          // 文件打开失败
    NO_MSG,             // 文件中无可修复的消息
    FIXING,             // 修复过程提示，反复刷新同一行
    FIX_DONE,           // 修复结束
    LESS_SPACE,         // 磁盘剩余小于1GB
    INVALID_PATH,       // 无效路径
    COMPRESS_ERROR,     // 压缩初始化失败
    DECOMPRESS_ERROR,   // 解压初始化失败
    COMPRESS_CHUNK_ERROR,     // 压缩chunk失败
    DECOMPRESS_CHUNK_ERROR,   // 解压chunk失败
    CLOSE_FAIL,         // 关闭bag文件失败
    RENAME_TO_BAG_ERROR,   // 重命名active文件失败
    MAX_TYPE
};

struct FixEchoInfo {
    FixEchoInfo();
    ara::core::String fileName;
    std::atomic<uint32_t> numFixed;
    std::atomic<uint32_t> numTotal;
};

struct FixOptions {
    FixOptions();
    ara::core::String fileName;
    ara::core::String outPath;
};

using PrintCallBack = std::function<void(FixPrintType, const FixEchoInfo&)>;
class RtfBagFix {
public:
    explicit RtfBagFix(const FixOptions& fixOptions);
    ~RtfBagFix();
    bool Fix();
    void Stop();
    void RegPrintCallback(const PrintCallBack& callback);
    double GetFixProgress() const;

protected:
    bool OpenRead();
    bool Run();
    bool DoRun(std::unique_ptr<RtfView>& viewPtr);
    bool HandleAcitveFile();
    bool WriteMsgEntityToBag(const RtfMsgEntity& msgEntity, RtfBagFile& bagFile) noexcept;
    bool CheckDisk();
    void PrintUserInfo(const FixPrintType& type, const FixEchoInfo& echoInfo) const;

private:
    PrintCallBack     printCallback_;
    ara::core::String fileName_;
    ara::core::String outputPath_;
    ara::core::String targetFileName_;
    ara::core::String writeFileName_;
    bool              isStop_;
    FixEchoInfo fixEchoInfo_;
    std::shared_ptr<RtfBagFile> bag_;
    uint32_t version_;
    rtf::rtfbag::FileCompressionType compressionType_;
    uint64_t compressErrorTime_;
};
}  // namespace rtfbag
}  // namespace rtf
#endif // RTF_PLAYER_H
