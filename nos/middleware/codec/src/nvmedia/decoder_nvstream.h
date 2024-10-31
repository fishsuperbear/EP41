/**
 * @file decoder_nvstream.h
 * @author zax (maxiaotian@hozonauto.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-12
 * 
 * Copyright Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * 
 */
#ifndef MIDDLEWARE_CODEC_SRC_ORIN_DECODER_NVSTREAM_H_
#define MIDDLEWARE_CODEC_SRC_ORIN_DECODER_NVSTREAM_H_

#include "codec/include/decoder.h"
#include "codec/src/nvmedia/ide/video_dec_ctx.h"

#include <memory>

class NvReplayer;

namespace hozon {
namespace netaos {
namespace codec {

class DecoderNvStream : public Decoder {
   public:
    DecoderNvStream();
    ~DecoderNvStream();

    /**
     * @brief DecoderNvStream init.
     * 
     * @param pic_infos camera record files info.
     * @return CodecErrc 
     */
    CodecErrc Init(const std::string& config_file);
    CodecErrc Init(const DecodeInitParam& init_param);
    CodecErrc Init(const PicInfos& pic_infos) override;
    // int Init(DecoderNvStreamCb DecoderNvStreamCb);

    /**
     * @brief 
     * 
     * @param in_buff 
     * @return CodecErrc 
     */
    CodecErrc Process(const DecodeBufInfo& info, const std::string& in_buff) override;
    int GetWidth();
    int GetHeight();
    int GetFormat();

   private:
    int CheckVersion(void);
    // int Decode(const std::string& in_buff, DecoderBufNvSpecific& out_buff);
    // int32_t cbBeginSequence(void* ptr, const NvMediaParserSeqInfo* pnvsi);
    // NvMediaStatus cbDecodePicture(void* ptr, NvMediaParserPictureData* pd);
    // NvMediaStatus cbDisplayPicture(void* ptr, NvMediaRefSurface* p, int64_t llPts);
    // void cbUnhandledNALU(void* ptr, const uint8_t* buf, int32_t size);
    // NvMediaStatus cbAllocPictureBuffer(void* ptr, NvMediaRefSurface** p);
    // int StreamVC1SimpleProfile(VideoDecCtx ctx);
    // void cbRelease(void* ptr, NvMediaRefSurface* p);
    // void cbAddRef(void* ptr, NvMediaRefSurface* p);
    // NvMediaStatus cbGetBackwardUpdates(void* ptr, NvMediaVP9BackwardUpdates* backwardUpdate);
    std::string filename;
    std::shared_ptr<NvReplayer> nvreplayer_;
};
}  // namespace codec
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_CODEC_SRC_ORIN_DECODER_NVSTREAM_H_
