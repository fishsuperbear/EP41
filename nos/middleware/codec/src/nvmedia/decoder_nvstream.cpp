#include "codec/src/nvmedia/decoder_nvstream.h"

#include <string.h>
#include <sys/stat.h>

#include "codec/src/codec_logger.h"
#include "codec/src/function_statistics.h"
#include "codec/src/nvmedia/utils/scibuf_utils.h"
#include "log_utils.h"
#include "nvmedia_common_encode_decode.h"
#include "sensor/nvs_producer/nv_replayer_api.h"

namespace hozon {
namespace netaos {
namespace codec {
enum LogLevelLogger {
    LEVEL_ERR = 0,
    LEVEL_WARN = 1,
    LEVEL_INFO = 2,
    LEVEL_DBG = 3,
};

DecoderNvStream::DecoderNvStream() : nvreplayer_(std::make_unique<NvReplayer>()) {}

DecoderNvStream::~DecoderNvStream() {
    nvreplayer_->CloseAllChannels();
}

CodecErrc DecoderNvStream::Init(const std::string& config_file) {
    return kDecodeNotImplemented;
}

CodecErrc DecoderNvStream::Init(const DecodeInitParam& init_param) {
    return kDecodeNotImplemented;
}

CodecErrc DecoderNvStream::Init(const PicInfos& pic_infos) {
    nvreplayer_->CreateProducerChannels(pic_infos);
    return kDecodeSuccess;
}

CodecErrc DecoderNvStream::Process(const DecodeBufInfo& info, const std::string& in_buff) {
    auto input = std::make_shared<InputBuf>();
    input->data = in_buff;
    input->frame_type = info.frame_type;
    input->post_time = info.post_time;
    nvreplayer_->Post(info.sid, input);
    return kDecodeSuccess;
}

int DecoderNvStream::GetWidth() {
    return kDecodeSuccess;
}

int DecoderNvStream::GetHeight() {
    return 0;
}

int DecoderNvStream::GetFormat() {
    return 0;
}

}  // namespace codec
}  // namespace netaos
}  // namespace hozon
