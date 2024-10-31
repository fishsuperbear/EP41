#include "codec_logger.h"

namespace hozon {
namespace netaos {
namespace codec {

class CodecLoggerInitializer {
   public:
    CodecLoggerInitializer() {
        CodecLogger::GetInstance().setLogLevel(static_cast<int32_t>(CodecLogger::CodecLogLevelType::INFO));
        CodecLogger::GetInstance().CreateLogger("CODEC");
        std::cout << "create logger.\n";
    }
};

}  // namespace codec
}  // namespace netaos
}  // namespace hozon
