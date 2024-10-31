#include "logger.h"

#define HELLO_LOG_FATAL logger.LogFatal()
#define HELLO_LOG_ERROR logger.LogError()
#define HELLO_LOG_WARN logger.LogWarn()
#define HELLO_LOG_INFO logger.LogInfo()
#define HELLO_LOG_DEBUG logger.LogDebug()
#define HELLO_LOG_VERBOSE logger.LogVerbose()

// #ifdef BUILD_FOR_X86
// #error "============ build for x86"
// #endif 

// #ifdef BUILD_FOR_J5
// #error "============ build for j5"
// #endif 

// #ifdef BUILD_FOR_MDC
// #error "============ build for mdc"
// #endif 

int main(int argc, char* argv[]) {
    hozon::netaos::log::EasyLogger logger;
    logger.Init(hozon::netaos::log::LogLevel::kInfo);

    HELLO_LOG_INFO << "build for " << TARGET_PLATFORM << ", type " << BUILD_TYPE;

    HELLO_LOG_FATAL << "Fatal log test.";
    HELLO_LOG_ERROR << "Errot log test.";
    HELLO_LOG_WARN << "Warn log test.";
    HELLO_LOG_INFO << "Info log test.";
    HELLO_LOG_DEBUG << "Debug log test, should never print this.";
    HELLO_LOG_VERBOSE << "Verbose log test, should never print this.";
}