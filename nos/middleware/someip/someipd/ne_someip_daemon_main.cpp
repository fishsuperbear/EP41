/**

* Copyright @ 2020 - 2027 iAuto Software(Shanghai) Co., Ltd.

* All Rights Reserved.

*

* Redistribution and use in source and binary forms, with or without

* modification, are NOT permitted except as agreed by

* iAuto Software(Shanghai) Co., Ltd.

*

* Unless required by applicable law or agreed to in writing, software

* distributed under the License is distributed on an "AS IS" BASIS,

* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

*/
#include <unistd.h>
#include <stdio.h>
#include <signal.h>
#include <string.h>
#include "cJSON.h"
#include "ne_someip_daemon.h"
#include "ne_someip_log.h"

#include "em/include/proctypes.h"
#include "em/include/exec_client.h"


#define NE_SOMEIP_CONFIG_FILE_MDC       ("/opt/usr/app/1/runtime_service/neta_someipd/conf/someipd_config.json")
#define NE_SOMEIP_CONFIG_FILE_J5        ("/userdata/runtime_service/neta_someipd/conf/someipd_config.json")
#define NE_SOMEIP_CONFIG_FILE_ORIN      ("/app/runtime_service/neta_someipd/conf/someipd_config.json")
#define NE_SOMEIP_CONFIG_FILE_DEFAULT   ("/app/runtime_service/neta_someipd/conf/someipd_config.json")

#define NE_SOMEIP_LOG_FILE_PATH         ("/opt/usr/log/soc_log/")
#define NE_SOMEIP_LOG_APP_NAME          ("someipd")
// 10M single log file
#define NE_SOMEIP_LOG_FILE_MAX_SIZE     (20*1024*1024)
// 10 log files，9 zip file and 1 log file
#define NE_SOMEIP_LOG_FILE_MAX_CNT      (10)

bool stop_flag = false;

void SigHandler(int signum)
{
    stop_flag = true;
}

void InitLog()
{
    char* config_file = NULL;
    char* app_name = NE_SOMEIP_LOG_APP_NAME;
    char* log_path = NE_SOMEIP_LOG_FILE_PATH;
    int   log_level = NE_SOMEIP_LOG_LEVEL_INFO;
    int   log_console = 0;
    int   max_files_count = NE_SOMEIP_LOG_FILE_MAX_CNT;
    int   max_file_size = NE_SOMEIP_LOG_FILE_MAX_SIZE;

#ifdef BUILD_FOR_MDC
    config_file = NE_SOMEIP_CONFIG_FILE_MDC;
#elif BUILD_FOR_J5
    config_file = NE_SOMEIP_CONFIG_FILE_J5;
#elif BUILD_FOR_ORIN
    config_file = NE_SOMEIP_CONFIG_FILE_ORIN;
#else
    config_file = NE_SOMEIP_CONFIG_FILE_DEFAULT;
#endif

    cJSON* root = NULL;
    if (!access(config_file, F_OK)) {
        FILE* file_fd = fopen(config_file, "rb");
        fseek(file_fd, 0, SEEK_END);
        long data_length = ftell(file_fd);
        rewind(file_fd);

        char* file_data = (char*)malloc(data_length + 1);
        memset(file_data, 0, data_length + 1);

        fread(file_data, 1, data_length, file_fd);
        fclose(file_fd);
        file_fd = NULL;

        root = cJSON_Parse(file_data);
        if (NULL != file_data) {
            free(file_data);
            file_data = NULL;
        }

        cJSON_GetObjectItem(root, "LogAppName") ? app_name = cJSON_GetStringValue(cJSON_GetObjectItem(root, "LogAppName"))
                                                : app_name = NE_SOMEIP_LOG_APP_NAME;
        cJSON_GetObjectItem(root, "LogLevel") ? log_level = (int)cJSON_GetNumberValue(cJSON_GetObjectItem(root, "LogLevel"))
                                              : log_level = NE_SOMEIP_LOG_LEVEL_INFO;
        cJSON_GetObjectItem(root, "LogConsole") ? log_console = (int)cJSON_GetNumberValue(cJSON_GetObjectItem(root, "LogConsole"))
                                                : log_console = 0;
        cJSON_GetObjectItem(root, "LogFilePath") ? log_path = cJSON_GetStringValue(cJSON_GetObjectItem(root, "LogFilePath"))
                                                 : log_path = NE_SOMEIP_LOG_FILE_PATH;
        cJSON_GetObjectItem(root, "MaxLogFileNum") ? max_files_count = (int)cJSON_GetNumberValue(cJSON_GetObjectItem(root, "MaxLogFileNum"))
                                                : max_files_count = NE_SOMEIP_LOG_FILE_MAX_CNT;
        cJSON_GetObjectItem(root, "MaxSizeOfLogFile") ? max_file_size = (int)cJSON_GetNumberValue(cJSON_GetObjectItem(root, "MaxSizeOfLogFile"))*1024*1024
                                                : max_file_size = NE_SOMEIP_LOG_FILE_MAX_SIZE;
    }

    // 限制log文件的大小为 5~20 M
    if (max_file_size < 5*1024*1024){
        max_file_size = 5*1024*1024;
    }
    if (max_file_size > 20*1024*1024) {
        max_file_size = 20*1024*1024;
    }

    // 读取环境变量是否临时设置log level和控制台打印
    char env_level[128] = { 0 };
    memset(env_level, 0x00, sizeof(env_level));
    strncpy(env_level, app_name, strlen(app_name));
    strncpy(&env_level[strlen(app_name)], "_LOG_LEVEL", strlen("_LOG_LEVEL"));
    if (getenv(env_level)) {
        if (0 != strtol(getenv(env_level), NULL, 10)) {
            log_level = strtol(getenv(env_level), NULL, 10);
        }
    }
    char env_console[128] = { 0 };
    memset(env_console, 0x00, sizeof(env_console));
    strncpy(env_console, app_name, strlen(app_name));
    strncpy(&env_console[strlen(app_name)], "_LOG_CONSOLE", strlen("_LOG_CONSOLE"));
    if (getenv(env_console)) {
        log_console = (strtol(getenv(env_console), NULL, 10) == 1) ? 1: 0;
    }

    ne_someip_log_init(app_name, log_level, log_console, log_path, max_files_count, max_file_size);

    if (NULL != root) {
        cJSON_Delete(root);
        root = NULL;
    }
}

int main()
{
    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);

    InitLog();
    ne_someip_daemon_t* daemon = ne_someip_daemon_init();
    ne_someip_log_info("someip daemon starting");

    if (NULL == daemon) {
        ne_someip_log_error("someip daemon init error");
        exit(1);
    }

    std::shared_ptr<hozon::netaos::em::ExecClient> execli = std::make_shared<hozon::netaos::em::ExecClient>();
    execli->ReportState(hozon::netaos::em::ExecutionState::kRunning);

    while (!stop_flag) {
        sleep(1);
    }

    ne_someip_log_info("someip daemon exiting");
    execli->ReportState(hozon::netaos::em::ExecutionState::kTerminating);
    ne_someip_daemon_deinit();
    ne_someip_log_deinit();
    exit(0);
}
/* EOF */
