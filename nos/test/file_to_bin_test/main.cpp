#include "update_manager/file_to_bin/file_to_bin.h"
#include "update_manager/log/update_manager_logger.h"



int main(int argc, char * argv[])
{
    hozon::netaos::update::UpdateManagerLogger::GetInstance().InitLogging("TO_BIN_TEST",    // the id of application
        "ecu update file transition to bin application", // the log id of application
        hozon::netaos::update::UpdateManagerLogger::UpdateLogLevelType::LOG_LEVEL_TRACE, //the log level of application
        hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE, //the output log mode
        "./", //the log file directory, active when output log to file
        10, //the max number log file , active when output log to file
        20 //the max size of each  log file , active when output log to file
    );
    hozon::netaos::update::UpdateManagerLogger::GetInstance().CreateLogger("ToBinTest");
    hozon::netaos::update::FileToBin fileToBinTestHdl;

    fileToBinTestHdl.Transition(argv[argc-2], argv[argc-1]);
    return 0;
}
