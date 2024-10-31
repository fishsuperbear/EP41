#ifndef DEFINE_H
#define DEFINE_H

#include "em/include/proctypes.h"

namespace hozon {
namespace netaos {
namespace em {

using namespace std;

enum class MapCompareType : uint32_t {
    DIFF_DEL = 0,
    DIFF_ADD = 1,
    DIFF_CPY = 2
};

enum class Sortord : uint32_t {
    ASCEND = 0,
    DESCEND = 1
};

enum class ModeState : uint32_t {
    DEFAULT = 0,
    STARTING = 1,
    FINISHED = 2
};


struct ModeOrder {
    std::string mode;
    uint32_t order;
};


#define DEF_MAX_CPU_CORE 12
#define PROC_STATE_DETECT_PERIOD 50     //ms
#define GROUP_STATE_DETECT_PERIOD 200   //ms
#define PROC_KEEP_ALIVE_PERIOD 200      //ms
#define PROC_RESTART_MAX_TIMES 4

/* mode definition */
#define OFF_MODE "Off"
#define ABNORMAL_MODE "Abnormal"
#define DEFAULT_MODE "Startup"
#define DESAY_UPDATE_SERVICE_NAME "svp_update"

#define PROC_CONFIG_FOLDER_NAME "etc"
#define PROC_CONFIG_FILE_NAME "MANIFEST.json"

#define PROCESS_DIR_PATH "/app/runtime_service"
#define MACHINE_MANIFEST_FILE "/app/conf/machine_manifest.json"
#define EXECMAGER_CONFIG_FILE "/app/conf/em_config.json"
#define STARTUP_MANIFEST_FILE "/cfg/conf_em/startup_manifest.json"
#define DESAY_UPDATE_SERVICE_SHELL "/app/scripts/svp_restart.sh"

/* param for debug and test */
#define DEV_ENVRION_PROC_DIR "EM_DEV_APP_DIR"
#define DEV_ENVRION_CONF_DIR "EM_DEV_CONF_DIR"
#define DEV_MACHINE_MANIFEST_FILE "machine_manifest.json"
#define DEV_STARTUP_MANIFEST_FILE "startup_manifest.json"
#define DEV_EXECMAGER_CONFIG_FILE "em_config.json"

const std::string proc_state[6] = {"IDLE", "STARTING", "RUNNING", "TERMINATING", "TERMINATED", "ABORTED"};
const std::string exec_state[3] = {"IDLE", "RUNNING", "TERMINATING"};

}}}
#endif
