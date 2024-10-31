#include "camera_venc_logger.h"

namespace hozon {
namespace netaos {
namespace cameravenc {

class CameraVencLoggerInitializer {
public:
    CameraVencLoggerInitializer() {
        CameraVencLogger::GetInstance().setLogLevel(
            static_cast<int32_t>(CameraVencLogger::CameraVencLogLevelType::INFO));
        CameraVencLogger::GetInstance().InitLogging(
            "CAMV",                                // the id of application
            "camera venc application",                           // the log id of application
            CameraVencLogger::CameraVencLogLevelType::INFO,  // the log
                                                    // level of
                                                    // application
            hozon::netaos::log::HZ_LOG2FILE,  // the output log mode
            "/opt/usr/log/soc_log/",  // the log file directory, active when output log to file
            10,    // the max number log file , active when output log to file
            20     // the max size of each  log file , active when output log to file
        );
        CameraVencLogger::GetInstance().CreateLogger("CAMV");
        std::cout << "create camera venc logger.\n";
    }
};

static CameraVencLoggerInitializer s_logger_initiliazer_;
}
}
}
