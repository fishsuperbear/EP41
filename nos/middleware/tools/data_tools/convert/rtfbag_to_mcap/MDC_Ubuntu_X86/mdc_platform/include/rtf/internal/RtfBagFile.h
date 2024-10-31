/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description:
 *      This file is the implement of class BagFile.
 *      BagFile will create a bag file and provide read or write operation
 * Create: 2019-11-30
 * Notes: NA
 */
#ifndef RTF_BAG_FILE_H
#define RTF_BAG_FILE_H

#include <iostream>
#include <mutex>
#include <set>
#include <sys/types.h>
#include <tuple>

#include "ara/core/map.h"
#include "ara/core/string.h"
#include "ara/core/vector.h"
#include "rtf/internal/compressor.h"
#include "rtf/internal/RtfBagStructs.h"
#include "rtf/internal/RtfBuffer.h"
#include "rtf/internal/RtfChunkManager.h"
#include "rtf/internal/RtfHeader.h"
#include "rtf/maintaind/impl_type_eventinfo.h"
#include "rtf/rtfbag/RtfBagInfo.h"

namespace rtf {
namespace rtfbag {
enum class FileCompressionType : uint32_t {
    NONE     = 0,    // 未压缩
    GZIP     = 1,    // GZIP
    ZLIB     = 2,    // ZLIB
};

enum class FileErrorCode : int32_t {
    SUCCESS     = 0,             // 成功
    WRITE_FAIL  = -1,            // 写失败
    READ_FAIL   = -2,            // 读失败
    COMPRESS_ERROR   = -3,       // 压缩失败
    DECOMPRESS_ERROR = -4,       // 解压失败
};

class RtfBagFile {
    friend class RtfView;
public:
    RtfBagFile();
    ~RtfBagFile();
    void CopyOtherBagFileReadResult(const RtfBagFile& other);
    /*
     * @Description: 打开文件并写入文件头相关信息
     * @param: input ara::core::String const& fileName
     * @return: 0: 成功 else: 失败
     */
    int32_t OpenWrite(ara::core::String const& fileName, ara::core::String const& filePath, ara::core::String const& file);

    /*
     * @Description: 关闭文件写入，此时会写入ConnetionInfo，ChunkInfo等信息
     * @param: none
     * @return: 0: 成功 else: 失败
     */
    FileErrorCode CloseWrite();

    FileErrorCode Write(rtf::rtfbag::EventMsg const& eventMsg, uint64_t const& time, uint8_t const *msgBuff, uint32_t buffLen);

    int32_t OpenRead(ara::core::String const& fileName);
    int32_t OpenReadActiveFile(ara::core::String const& fileName);
    bool CloseRead();
    FileErrorCode ReadMsgDataIntoStream(MessageIndex const& msgIndex, RtfBuffer& buffer, const std::uint32_t len = 0);
    bool ReadMsgDataSize(MessageIndex const& msgIndex, uint32_t& size) const;
    /*
     * @Description: 获取当前bag文件的文件大小
     * @param: none
     * @return: uint64_t 文件大小
     */
    uint64_t GetFileSize() const;

    /*
     * @Description: 获取当前打开文件的文件名称
     * @param: none
     * @return: ara::core::String 文件名称
     */
    ara::core::String GetFileName() const;
    void SetMaxChunkSize(uint32_t chunkSize);

    /*
     * @Description: 获取bag文件头信息
     * @param:
     * @return: true: 成功 false： 失败
     */
    bool GetFileHeaderInfo(BagFileHeadInfo& headerInfo);

    void SetStartTime();
    void SetStopTime();
    bool CheckIsSameBag(const RtfBagFile &other);
    bool OpenToReadBagFile(ara::core::String const& fileName);
    void SetVersion(const uint32_t& version);
    uint32_t GetVersion() const;
    FILE* file_;
    void SetCompressionType(const FileCompressionType& compressionType);
    FileCompressionType GetCompressionType() const;
private:
    int32_t InitCompressor(const bool& isWrite);
    bool CloseFile();
    // write chunk
    bool StartChunkWriting(uint64_t time);
    FileErrorCode StopChunkWriting();
    void ClearChunkInfo();
    void ClearCurChunkNewEvent();
    FileErrorCode TransferChunkToStream();

    void ReleaseBuf(uint8_t* buf, const uint32_t& buflength, const ara::core::String& str) const;
    bool WriteChunk();
    bool WriteChunkForMsgs(const OutChunkHeader& chunkHeader,
                           std::queue<OutMessage>& queueBuff) const;
    bool WriteChunkForMsgsForCompress(const OutChunkHeader& chunkHeader);
    bool WriteChunkForChunkIndex(std::queue<OutMessage>& queueChunkIndex) const;
    void ResetBagFile();
    // write record
    bool WriteVersion() const;
    bool WriteFileHeaderRecord();
    void SetFileHeaderFieldName(ara::core::Vector<ara::core::String>& fieldNameString) const;
    bool WriteRecordHeader(ara::core::Vector<ara::core::String> const& fieldName,
        ara::core::Vector<ara::core::String> const& fieldValue) const;
    bool WriteChunkInfoRecords() const;
    bool WriteChunkInfoRecord(ChunkInfo chunkInfoRecord) const;
    bool WriteTempChunkInfoRecord(const ChunkInfo& chunkInfoRecord);
    bool WriteConnectionRecords() const;
    bool WriteConnectionRecord(const Connection& connRecord) const;
    bool WriteEventInfo(ara::core::Vector<ara::core::String>& fieldValueString,
                        const Connection& connRecord) const;
    bool WriteSerializationConfig(ara::core::Vector<ara::core::String>& fieldValueString,
                                  const Connection& connRecord) const;
    bool WriteTempConnectionRecord(const Connection& connRecord);
    bool WriteChunkHeaderRecord(uint32_t currChunkSize, uint32_t currChunkCompressedSize);
    bool WriteMessageRecord(uint32_t connnectionId,
        uint64_t time, uint8_t const *msgBuff, uint32_t msgLen);
    bool WriteChunkIndexRecords();
    bool WriteChunkIndexRecord(uint32_t id, const std::multiset<MessageIndex> &indexSet);
    bool TransferMessageIndexToBuff(uint8_t* pBuff, const uint32_t& dataLen,
                                    const std::multiset<MessageIndex> &indexSet) const;
    bool WriteChunkEndInfoRecord();
    uint32_t GetRecordHeaderLen(ara::core::Vector<ara::core::String> const& fieldName,
        ara::core::Vector<ara::core::String> const& fieldValue) const;
    uint32_t GetConnectionId(rtf::rtfbag::EventMsg const& eventMsg);

    // read record
    std::tuple<bool, ino_t, time_t> GetBagChangeInfo(const ara::core::String &realPath);
    bool StartReading();
    bool StartReadChunkIndexRecord();
    bool ReadVersion();
    bool ReadFileHeaderRecord();
    bool ReadConnectionRecord();
    bool ReadConnectionFeild(const ReadMap &readMap, Connection &connection);
    bool ReadConnectionFeildAppend(const ReadMap &readMap, Connection &connection) const;
    bool ReadConnectionEventInfo(const ReadMap &readMap, Connection &connection) const;
    bool ReadConnectionSerializeConfig(const ReadMap &readMap, Connection &connection) const;

    bool ReadChunkInfoRecord();
    bool ReadChunkIndexRecord();
    bool ReadChunkHeader(ChunkHeader& chunkHeader) const;
    bool OptionMatch(const ReadMap& readMap, const uint8_t& opt) const;
    // read header
    template<typename T>
    bool ReadField(const ReadMap& readMap, const ara::core::String& field, T* data) const;
    bool ReadField(const ReadMap& readMap, const ara::core::String& field, ara::core::String& data) const;
    bool ReadHeader(RtfHeader& header) const;
    bool ReadHeaderFromStream(RtfHeader& header, const uint8_t* buff) const;
    FileErrorCode ReadMsgDataFromBuffer(RtfBuffer& buffer, const uint64_t& offset, const std::uint32_t len) const;
    FileErrorCode ReadMsgDataFromStream(RtfBuffer& buffer, MessageIndex const& msgIndex, const std::uint32_t len);
    FileErrorCode DecompressChunkToStream(MessageIndex const& msgIndex);
    void UpdateVersion(BagFileHeadInfo& headerInfo) const;

    bool CheckSystemIsAos() const;
    void GetFileHeaderRecordTimeInfo(BagFileHeadInfo& headerInfo) const;
    bool StartReadingActiveFile();
    bool ReadConnectionRecordFromActive();
    bool ReadChunkInfoRecordFromActive();
    bool ReadChunkEndInfo(const ChunkHeader& chunkHeader);
    void SetBagFileMessageInfo(BagFileMessageInfo &msgInfo, const Connection& conn);
    void GetInfoFromOldBag(BagFileHeadInfo& headerInfo);
    void GetInfoFromOptimizedBag(BagFileHeadInfo& headerInfo);
    void SetEventSumInfo();
    void SetChunkConSumInfo(ara::core::String const& eventName, uint64_t const& time);
    bool WriteConnectionRecordFieldName(ara::core::Vector<ara::core::String>& fieldNameStr) const;
    bool WriteConnectionRecordFieldValue(ara::core::Vector<ara::core::String>& fieldValueStr,
                                         const Connection& connection) const;
    bool OpenReadForBagHeaderInfo(ara::core::String const& fileName);

    ara::core::String fileName_;
    uint32_t maxChunkSize_;
    uint64_t fileSize_;
    uint32_t chunkCount_;
    uint32_t connCount_;
    uint64_t connPos_;
    uint64_t nextChunkPos_;
    uint64_t chunkInfoRecordPos_;
    uint32_t chunkStatus_;
    BagFileHeader bagFileHeader_;
    uint64_t bagFileHeaderPos_;
    uint32_t version_;
    uint32_t writeVersion_;
    FileCompressionType compressionType_;
    FileCompressionType writeCompressionType_;
    uint32_t compressedSize_;
    ara::core::String bagVersion_;  // current bag version
    bool fileWriteOpen_;
    bool fileReadOpen_;

    // Current chunk
    bool chunkOpen_;
    uint64_t currChunkDataStartPos_;
    uint64_t currChunkDataEndPos_;
    ChunkInfo currChunkInfo_;
    ara::core::Vector<ChunkInfo> chunks_;

    uint64_t currentReadChunkDataPos_;
    uint8_t* curRdChunkBuff_;
    ara::core::Map<ara::core::String, uint32_t> eventConnectionId_;
    std::set<ara::core::String> eventRecorded_;
    ara::core::Map<ara::core::String, Connection> eventConnection_;
    ara::core::Map<ara::core::String, ChunkConSumInfo> eventConInChunk_;
    ara::core::Map<uint32_t, Connection> connections_;
    uint32_t currentChunkNewEventCount_;
    uint32_t currentChunkConnectionCount_;

    ara::core::Map<uint32_t, std::multiset<MessageIndex>> connectionIdIndex_;
    ara::core::Map<uint32_t, std::multiset<MessageIndex>> currChunkIndex_;

    uint64_t startRecordRealTime_;
    uint64_t startRecordVirtualTime_;
    uint64_t stopRecordRealTime_;
    uint64_t stopRecordVirtualTime_;
    mutable std::mutex multiRead_;
    std::tuple<bool, ino_t, time_t> bagChangInfo_;
    std::shared_ptr<rtf::RtfChunkManager> RtfChunkManager_;
    const uint32_t bagVer110 = 110;
    const uint32_t bagVer120 = 120;
    const uint32_t bagVer130 = 130;
    const uint32_t bagVer140 = 140;
    const uint32_t bagVer150 = 150;
    const uint32_t bagVer160 = 160;
    const uint32_t bagVer170 = 170;
    const uint32_t bagVer180 = 180;
    const uint32_t bagVer190 = 190;
    const uint32_t bagVer210 = 210;
    std::shared_ptr<mdc::ddt::Compressor> processor_;
};
}  // namespace rtfbag
}  // namespace rtf
#endif  // RTF_BAG_FILE_H
