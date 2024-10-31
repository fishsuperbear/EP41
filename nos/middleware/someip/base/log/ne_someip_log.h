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
#ifndef BASE_LOG_NE_SOMEIP_LOG_H_
#define BASE_LOG_NE_SOMEIP_LOG_H_

#ifdef  __cplusplus
extern "C" {
#endif

// define someip log level
typedef enum NE_SOMEIP_LOG_LEVEL {
    NE_SOMEIP_LOG_LEVEL_VERBOSE = 0,
    NE_SOMEIP_LOG_LEVEL_DEBUG = 1,
    NE_SOMEIP_LOG_LEVEL_INFO = 2,
    NE_SOMEIP_LOG_LEVEL_WARNING = 3,
    NE_SOMEIP_LOG_LEVEL_ERROR = 4,
    NE_SOMEIP_LOG_LEVEL_FATAL = 5,
    NE_SOMEIP_LOG_LEVEL_NOLOG = 6,
    NE_SOMEIP_LOG_LEVEL_CONSOLE = 0xFF,
} ne_someip_log_level_t;


void ne_someip_log_output(ne_someip_log_level_t level, const char* function, const char* file, int line, const char* fmt, ...);

void ne_someip_log_init(char* app_name, int log_level, int log_console, char* log_path, int max_files_count, int max_file_size);
void ne_someip_log_deinit();

// someip
#define ne_someip_log_fatal(...)  \
        ne_someip_log_output(NE_SOMEIP_LOG_LEVEL_FATAL, __FUNCTION__, __FILE__, __LINE__, __VA_ARGS__);

#define ne_someip_log_error(...)  \
        ne_someip_log_output(NE_SOMEIP_LOG_LEVEL_ERROR, __FUNCTION__, __FILE__, __LINE__, __VA_ARGS__);

#define ne_someip_log_warn(...)  \
        ne_someip_log_output(NE_SOMEIP_LOG_LEVEL_WARNING, __FUNCTION__, __FILE__, __LINE__, __VA_ARGS__);

#define ne_someip_log_info(...)  \
        ne_someip_log_output(NE_SOMEIP_LOG_LEVEL_INFO, __FUNCTION__, __FILE__, __LINE__, __VA_ARGS__);

#define ne_someip_log_debug(...)  \
        ne_someip_log_output(NE_SOMEIP_LOG_LEVEL_DEBUG, __FUNCTION__, __FILE__, __LINE__, __VA_ARGS__);

#define ne_someip_log_verbose(...)  \
        ne_someip_log_output(NE_SOMEIP_LOG_LEVEL_VERBOSE, __FUNCTION__, __FILE__, __LINE__, __VA_ARGS__);

#define ne_someip_log_nlog(...)  \
        ne_someip_log_output(NE_SOMEIP_LOG_LEVEL_NOLOG, __FUNCTION__, __FILE__, __LINE__, __VA_ARGS__);

#ifdef  __cplusplus
}
#endif
#endif // BASE_LOG_NE_SOMEIP_LOG_H_
/* EOF */
