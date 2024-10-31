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

#include <stdio.h>
#include <stdarg.h>
#include <syslog.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include <pthread.h>

#if ( defined linux ) || ( defined __linux__ )
#include <sys/syscall.h>
#endif

#include "ne_someip_log.h"

#define MAX_LOG_SIZE 1024
#define NE_SOMEIP_FILE_NAME(x) strrchr(x, '/') ? strrchr(x, '/') + 1 : x

#define NE_LOG_FILE_FORMAT  ("_%04d_%04d-%02d-%02d_%02d-%02d-%02d.%s")
#define NE_SOMEIP_LOG_FORMAT ("[%s] [%s] [%s] [%d %ld %s@%s(%d) | %s]\n")

static char log_level_str[8][10] = { "verbose", "debug", "info", "warn", "error",  "fatal", "nlog", "unknown" };


typedef struct
{
    /* data */
    FILE* pfile;
    int cur_files_count;
    int cur_file_index;
    int cur_file_size;
    char cur_file_name[64];
    char g_app_name[16];
    char g_log_path[64];
    int g_log_level;
    int g_log_console;
    int g_max_files_count;
    int g_max_file_size;
    pthread_mutex_t cur_mutex;
} ne_someip_log_file;

// 默认单个log文件最大10M， 最多10个log文件
static ne_someip_log_file g_logfile = { NULL, 0, 0, 0,
    { 0 }, { 0 }, { 0 }, 0, 0, 0, 0, PTHREAD_MUTEX_INITIALIZER };


static pid_t ne_someip_log_gettid()
{
#if ( defined linux ) || ( defined __linux__ )
    return syscall( __NR_gettid );
#else
    return gettid();
#endif
}

// 比较函数，用于排序
int compare_str(const void* a, const void* b) {
    return strcmp(*(const char**)a, *(const char**)b);
}

int compare_int(const void* a, const void* b) {
    return (*(const int*)a) - (*(const int*)b);
}

void ne_someip_log_init(char* app_name, int log_level, int log_console, char* log_path, int max_files_count, int max_file_size)
{
    if (NULL == app_name) {
        return;
    }
    memset(g_logfile.g_app_name, 0x00, sizeof(g_logfile.g_app_name));
    memcpy(g_logfile.g_app_name, app_name, (strlen(app_name) < (sizeof(g_logfile.g_app_name) - 1)
        ? strlen(app_name) : (sizeof(g_logfile.g_app_name) - 1)));
    memset(g_logfile.g_log_path, 0x00, sizeof(g_logfile.g_log_path));
    memcpy(g_logfile.g_log_path, log_path, (strlen(log_path) < (sizeof(g_logfile.g_log_path) - 1)
        ? strlen(log_path) : (sizeof(g_logfile.g_log_path) - 1)));
    g_logfile.g_log_level = log_level;
    g_logfile.g_log_console = log_console;
    g_logfile.g_max_file_size = max_file_size;
    g_logfile.g_max_files_count = max_files_count;

    // 初始化待写的文件指针
    struct tm dateinfo;
    struct stat filestat;
    struct dirent *entry = NULL;
    char filetype[16] = { 0 };
    char filename[128] = { 0 };
    char fileformat[128] = { 0 };
    char filessort[20][64] = { 0 };
    int indexsort[20] = { 0 };
    int filescount = 0;
    int index = 0;
    memset(filessort, 0x00, sizeof(filessort));
    DIR *plogdir = opendir(log_path);
    if (NULL == plogdir) {
        return;
    }
    memcpy(fileformat, app_name, strlen(app_name));
    memcpy(fileformat + strlen(app_name), NE_LOG_FILE_FORMAT, strlen(NE_LOG_FILE_FORMAT));

    pthread_mutex_init(&g_logfile.cur_mutex, NULL);
    pthread_mutex_lock(&g_logfile.cur_mutex);
    while ((entry = readdir(plogdir)) != NULL) {
        // 普通文件
        if (entry->d_type == DT_REG) {

            // 不是匹配的log格式的文件
            if (8 != sscanf(entry->d_name, fileformat, &index, &dateinfo.tm_year, &dateinfo.tm_mon, &dateinfo.tm_mday,
                &dateinfo.tm_hour, &dateinfo.tm_min, &dateinfo.tm_sec, filetype)) {
                continue;
            }

            // 如果不是匹配格式的后缀 log和zip的文件
            // if (0 != strcmp(filetype, "log") && 0 != strcmp(filetype, "zip")) {
            //     continue;
            // }

            memcpy(filessort[g_logfile.cur_files_count],  entry->d_name, strlen(entry->d_name));
            indexsort[g_logfile.cur_files_count] = index;
            ++g_logfile.cur_files_count;

            // .log通常是最新的log文件
            if (0 == strcmp(filetype, "log")) {
                memset(filename, 0x00, sizeof(filename));
                // 缝合目录和文件名形成完整文件路径
                memcpy(filename, log_path, strlen(log_path));
                memcpy(filename + strlen(log_path), entry->d_name, strlen(entry->d_name));

                stat(filename, &filestat);
                g_logfile.cur_file_index = index;
                memcpy(g_logfile.cur_file_name,  entry->d_name, strlen(entry->d_name));

                g_logfile.cur_file_size = filestat.st_size;
                g_logfile.pfile = fopen(filename, "a+");
            }
        }
    }

    // 关闭打开的文件目录
    closedir(plogdir);
    plogdir = NULL;
    filescount = g_logfile.cur_files_count;

    // 如果没有log文件, 则主动创建log文件
    // 当前没有相关的log文件，直接创建第一个
    if (NULL == g_logfile.pfile && g_logfile.cur_files_count == 0) {
        time_t rawtime;
        time(&rawtime);
        struct tm* timeinfo = localtime(&rawtime);
        g_logfile.cur_file_index = 1;
        g_logfile.cur_file_size = 0;
        memset(g_logfile.cur_file_name, 0x00, sizeof(g_logfile.cur_file_name));
        snprintf(g_logfile.cur_file_name, sizeof(g_logfile.cur_file_name), fileformat, g_logfile.cur_file_index,
            timeinfo->tm_year + 1900, timeinfo->tm_mon + 1, timeinfo->tm_mday, timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec, "log");

        // 缝合目录和文件名形成完整文件路径
        memset(filename, 0x00, sizeof(filename));
        memcpy(filename, log_path, strlen(log_path));
        memcpy(filename + strlen(log_path), g_logfile.cur_file_name, strlen(g_logfile.cur_file_name));
        g_logfile.pfile = fopen(filename, "a+");
        ++g_logfile.cur_files_count;
        pthread_mutex_unlock(&g_logfile.cur_mutex);
        return;
    }

    // 已经找到并打开了.log文件, 且log文件没有超过数量
    if (NULL != g_logfile.pfile && g_logfile.cur_files_count <= g_logfile.g_max_files_count) {
        pthread_mutex_unlock(&g_logfile.cur_mutex);
        return;
    }

    // 如果文件数量大于限制，需要删除旧的log文件，先根据文件名排序然后再执行删除操作
    // log文件序号排序
    qsort(indexsort, g_logfile.cur_files_count, sizeof(int), compare_int);
    // log文件同步排序
    // qsort(filessort, g_logfile.cur_files_count, sizeof(filessort[0]), compare_str);

    // 已经找到并打开了.log文件, 对应的log文件超过了限制数量，需要删掉旧的log文件
    if (NULL != g_logfile.pfile && g_logfile.cur_files_count > g_logfile.g_max_files_count) {
        int removecnt = g_logfile.cur_files_count - g_logfile.g_max_files_count;
        // 当前log文件是最大序号的文件, 删除最前面的最旧的文件
        if (g_logfile.cur_file_index == indexsort[g_logfile.cur_files_count - 1]) {
            for (int i = 0; i < removecnt; ++i) {
                for (int j = 0; j < filescount; ++j) {
                    if (indexsort[i] == strtol(filessort[j] + (strlen(app_name) + strlen("_")), NULL, 10)) {
                        // 缝合目录和文件名形成完整文件路径
                        memset(filename, 0x00, sizeof(filename));
                        memcpy(filename, log_path, strlen(log_path));
                        memcpy(filename + strlen(log_path), filessort[j], strlen(filessort[j]));
                        // 删除旧的文件
                        remove(filename);
                        --g_logfile.cur_files_count;
                    }
                }
            }
        }
        else {
            // .log文件序号经过了翻转
            int orderbefore = 0;
            int orderafter = 0;
            int removebefore = 0;
            int removeafter = 0;
            for (int i = 0; i < g_logfile.cur_files_count; ++i) {
                (indexsort[i] <= g_logfile.cur_file_index)  ? ++orderbefore : ++orderafter;
            }

            // 汇总.log前后需要删除的文件个数
            if (orderafter > removecnt ) {
                removeafter = removecnt;
                removebefore = 0;
            }
            else {
                removeafter = orderafter;
                removebefore = removecnt - orderafter;
            }

            for (int i = 0; i < filescount; ++i) {
                if (removebefore > 0 && indexsort[i] < g_logfile.cur_file_index) {
                    for (int j = 0; j < filescount; ++j) {
                        if (indexsort[i] == strtol(filessort[j] + (strlen(app_name) + strlen("_")), NULL, 10)) {
                            // 缝合目录和文件名形成完整文件路径
                            memset(filename, 0x00, sizeof(filename));
                            memcpy(filename, log_path, strlen(log_path));
                            memcpy(filename + strlen(log_path), filessort[j], strlen(filessort[j]));
                            // 删除旧的文件
                            remove(filename);
                            --removebefore;
                            --g_logfile.cur_files_count;
                        }
                    }
                }

                if (removeafter > 0 && indexsort[i] > g_logfile.cur_file_index) {
                    for (int j = 0; j < filescount; ++j) {
                        if (indexsort[i] == strtol(filessort[j] + (strlen(app_name) + strlen("_")), NULL, 10)) {
                            // 缝合目录和文件名形成完整文件路径
                            memset(filename, 0x00, sizeof(filename));
                            memcpy(filename, log_path, strlen(log_path));
                            memcpy(filename + strlen(log_path), filessort[j], strlen(filessort[j]));
                            // 删除旧的文件
                            remove(filename);
                            --removeafter;
                            --g_logfile.cur_files_count;
                        }
                    }
                }
            }
        }
        pthread_mutex_unlock(&g_logfile.cur_mutex);
        return;
    }

    // .log文件没有，可能被删除了，需要接着序号继续创建.log文件
    if (NULL == g_logfile.pfile && g_logfile.cur_files_count > 0) {
        // 最大的序号是否需要翻转
        for (int i = 0; i < g_logfile.cur_files_count; ++i) {
            if ((indexsort[i] + 1000 < indexsort[i + 1] && i < g_logfile.cur_files_count - 1) || (i == g_logfile.cur_files_count - 1)) {
                // 序号跳跃幅度大于100，则当前的即认为是最新的文件

                time_t rawtime;
                time(&rawtime);
                struct tm* timeinfo = localtime(&rawtime);
                g_logfile.cur_file_index = indexsort[i] + 1;
                if (g_logfile.cur_file_index > 9999) {
                    g_logfile.cur_file_index = 1;
                }
                g_logfile.cur_file_size = 0;
                memset(g_logfile.cur_file_name, 0x00, sizeof(g_logfile.cur_file_name));
                snprintf(g_logfile.cur_file_name, sizeof(g_logfile.cur_file_name), fileformat, g_logfile.cur_file_index,
                    timeinfo->tm_year + 1900, timeinfo->tm_mon + 1, timeinfo->tm_mday, timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec, "log");

                // 缝合目录和文件名形成完整文件路径
                memset(filename, 0x00, sizeof(filename));
                memcpy(filename, log_path, strlen(log_path));
                memcpy(filename + strlen(log_path), g_logfile.cur_file_name, strlen(g_logfile.cur_file_name));
                g_logfile.pfile = fopen(filename, "a+");
                ++g_logfile.cur_files_count;
                break;
            }
        }

        // 文件数量已经超过上限，需要删除旧文件
        if (g_logfile.cur_files_count > g_logfile.g_max_files_count) {
            // .log文件序号经过了翻转
            int orderbefore = 0;
            int orderafter = 0;
            int removebefore = 0;
            int removeafter = 0;
            int removecnt = g_logfile.cur_files_count - g_logfile.g_max_files_count;
            // 当前新建的文件尚未加入到排序队列中，文件数量需要减一
            for (int i = 0; i < g_logfile.cur_files_count - 1; ++i) {
                (indexsort[i] <= g_logfile.cur_file_index)  ? ++orderbefore : ++orderafter;
            }

            // 汇总.log前后需要删除的文件个数
            if (orderafter > removecnt ) {
                removeafter = removecnt;
                removebefore = 0;
            }
            else {
                removeafter = orderafter;
                removebefore = removecnt - orderafter;
            }

            // 遍历按照书序删除文件
            for (int i = 0; i < filescount; ++i) {
                if (removebefore > 0 && indexsort[i] < g_logfile.cur_file_index) {
                    for (int j = 0; j < filescount; ++j) {
                        if (indexsort[i] == strtol(filessort[j] + (strlen(app_name) + strlen("_")), NULL, 10)) {
                            // 缝合目录和文件名形成完整文件路径
                            memset(filename, 0x00, sizeof(filename));
                            memcpy(filename, log_path, strlen(log_path));
                            memcpy(filename + strlen(log_path), filessort[j], strlen(filessort[j]));

                            // 删除旧的文件
                            remove(filename);
                            --removebefore;
                            --g_logfile.cur_files_count;
                        }
                    }
                }

                if (removeafter > 0 && indexsort[i] > g_logfile.cur_file_index) {
                    for (int j = 0; j < filescount; ++j) {
                        if (indexsort[i] == strtol(filessort[j] + (strlen(app_name) + strlen("_")), NULL, 10)) {
                            // 缝合目录和文件名形成完整文件路径
                            memset(filename, 0x00, sizeof(filename));
                            memcpy(filename, log_path, strlen(log_path));
                            memcpy(filename + strlen(log_path), filessort[j], strlen(filessort[j]));

                            // 删除旧的文件
                            remove(filename);
                            --removeafter;
                            --g_logfile.cur_files_count;
                        }
                    }
                }
            }
        }
    }
    pthread_mutex_unlock(&g_logfile.cur_mutex);
}

void ne_someip_log_write(char* buff, int size)
{
    char filetype[16] = { 0 };      // 文件名后缀
    char filename[512] = { 0 };     // 文件名全路径
    char filerename[512] = { 0 };   // 重命名文件名
    char zipfilename[512] = { 0 };  // 压缩文件名

    // 初始当前log未开打开状态，没有初始化则不进行写文件操作
    if (g_logfile.pfile == NULL) {
        return;
    }

    pthread_mutex_lock(&g_logfile.cur_mutex);
    // 要写的log大小未超过单个文件的上限
    if (size + g_logfile.cur_file_size < g_logfile.g_max_file_size) {
        fwrite(buff, size, 1, g_logfile.pfile);
        fflush(g_logfile.pfile);
        g_logfile.cur_file_size += size;
        pthread_mutex_unlock(&g_logfile.cur_mutex);
        return;
    }

    // 要写的log大小已经超过单个文件的上限，写完当前文件后压缩，然后需要另写文件，同时若超过了文件总数还需要删除旧的log文件
    // 1. 完成当前文件满写操作后关闭该文件
    int lena = fwrite(buff, g_logfile.g_max_file_size - g_logfile.cur_file_size, 1, g_logfile.pfile);
    fflush(g_logfile.pfile);
    fclose(g_logfile.pfile);
    g_logfile.pfile = NULL;/*  */
    // 剩余的buff 大小记录到也就是下一个文件的待写的大小，等待新建文件后写入
    g_logfile.cur_file_size = size + g_logfile.cur_file_size - g_logfile.g_max_file_size;

    // 2.当前文件重命名，添加"_"后缀
    memset(filename, 0x00, sizeof(filename));
    memcpy(filename, g_logfile.g_log_path, strlen(g_logfile.g_log_path));
    memcpy(filename + strlen(g_logfile.g_log_path), g_logfile.cur_file_name, strlen(g_logfile.cur_file_name));
    memcpy(filerename, filename, strlen(filename));
    memcpy(filerename + strlen(filename), "_", strlen("_"));
    rename(filename, filerename);

    // 3.压缩已经写满的log文件
    memcpy(zipfilename, filename, strlen(filename) - strlen("log"));
    memcpy(zipfilename + strlen(zipfilename), "zip", strlen("zip"));
    char systemcmd[128] = { 0 };
    snprintf(systemcmd, sizeof(systemcmd), "zip %s %s >> /dev/null ", zipfilename, filerename);
    system(systemcmd);
    remove(filerename);

    // zipFile zipfile = zipOpen64(zipfilename, APPEND_STATUS_ADDINZIP);
    // zipOpenNewFileInZip64(zipfile,
    //                       filerename,
    //                       NULL,
    //                       NULL,
    //                       0,
    //                       NULL,
    //                       0,
    //                       NULL /* comment*/,
    //                       0,
    //                       Z_BEST_SPEED,
    //                       0);
    // char* zipbuf = malloc(g_logfile.cur_file_size + 1);
    // FILE* zipfp = fopen(filerename, "r");
    // fread(zipbuf, 1, sizeof(g_logfile.cur_file_size), zipfp);
    // zipWriteInFileInZip(zipfile, zipbuf, g_logfile.cur_file_size);
    // zipCloseFileInZip(zipfile);
    // zipClose_64(zipfile, NULL);
    // fclose(zipfp);
    // zipfp = NULL;
    // free(zipbuf);
    // remove(filerename);

    // 4.将剩余的信息写新的log文件
    ++g_logfile.cur_file_index;
    // 超过9999后序号需要翻转继续从1开始
    if (g_logfile.cur_file_index > 9999) {
        g_logfile.cur_file_index = 1;
    }
    // 生成log格式的文件名
    time_t rawtime;
    time(&rawtime);
    struct tm* timeinfo = localtime(&rawtime);
    char fileformat[128] = { 0 };
    memcpy(fileformat, g_logfile.g_app_name, strlen(g_logfile.g_app_name));
    memcpy(fileformat + strlen(g_logfile.g_app_name), NE_LOG_FILE_FORMAT, strlen(NE_LOG_FILE_FORMAT));
    memset(g_logfile.cur_file_name, 0x00, sizeof(g_logfile.cur_file_name));
    snprintf(g_logfile.cur_file_name, sizeof(g_logfile.cur_file_name), fileformat, g_logfile.cur_file_index,
        timeinfo->tm_year + 1900, timeinfo->tm_mon + 1, timeinfo->tm_mday, timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec, "log");
    // 缝合目录和文件名形成完整文件路径
    memset(filename, 0x00, sizeof(filename));
    memcpy(filename, g_logfile.g_log_path, strlen(g_logfile.g_log_path));
    memcpy(filename + strlen(g_logfile.g_log_path), g_logfile.cur_file_name, strlen(g_logfile.cur_file_name));
    g_logfile.pfile = fopen(filename, "a+");
    int lens = fwrite(buff + (size - g_logfile.cur_file_size), g_logfile.cur_file_size, 1, g_logfile.pfile);
    fflush(g_logfile.pfile);
    ++g_logfile.cur_files_count;

    // 5. 生成新的log文件后数量超过了文件上限 则清除旧的log文件
    if (g_logfile.cur_files_count > g_logfile.g_max_files_count) {
        // 初始化待写的文件指针
        struct tm dateinfo;
        struct stat filestat;
        struct dirent *entry = NULL;
        char appname[16] = { 0 };
        char filetype[16] = { 0 };
        char filename[128] = { 0 };
        char filessort[20][64] = { 0 };
        int indexsort[20] = { 0 };
        int index = 0;
        int cur_files_count = 0;
        memset(filessort, 0x00, sizeof(filessort));
        DIR *plogdir = opendir(g_logfile.g_log_path);
        while ((entry = readdir(plogdir)) != NULL) {
            // 普通文件
            if (entry->d_type == DT_REG) {

                // 不是匹配的log格式的文件
                if (8 != sscanf(entry->d_name, fileformat, &index, &dateinfo.tm_year, &dateinfo.tm_mon, &dateinfo.tm_mday,
                    &dateinfo.tm_hour, &dateinfo.tm_min, &dateinfo.tm_sec, filetype)) {
                    continue;
                }

                // 如果不是匹配格式的后缀 log和zip的文件
                // if (0 != strcmp(filetype, "log") && 0 != strcmp(filetype, "zip")) {
                //     continue;
                // }

                memcpy(filessort[cur_files_count],  entry->d_name, strlen(entry->d_name));
                indexsort[cur_files_count] = index;
                ++cur_files_count;
            }
        }
        // 关闭打开的文件目录
        closedir(plogdir);
        plogdir = NULL;

        // 如果文件数量大于限制，需要删除旧的log文件，先根据文件名排序然后再执行删除操作
        // log文件序号排序
        qsort(indexsort, g_logfile.cur_files_count, sizeof(int), compare_int);
        // log文件同步排序
        // qsort(filessort, g_logfile.cur_files_count, sizeof(filessort[0]), compare_str);

        // .log文件序号经过了翻转
        int orderbefore = 0;
        int orderafter = 0;
        int removebefore = 0;
        int removeafter = 0;
        int removecnt = g_logfile.cur_files_count - g_logfile.g_max_files_count;
        // 当前新建的文件尚未加入到排序队列中，文件数量需要减一
        for (int i = 0; i < g_logfile.cur_files_count; ++i) {
            (indexsort[i] <= g_logfile.cur_file_index)  ? ++orderbefore : ++orderafter;
        }

        // 汇总.log前后需要删除的文件个数
        if (orderafter > removecnt ) {
            removeafter = removecnt;
            removebefore = 0;
        }
        else {
            removeafter = orderafter;
            removebefore = removecnt - orderafter;
        }

        // 遍历按照书序删除文件
        for (int i = 0; i < cur_files_count; ++i) {
            if (removebefore > 0 && indexsort[i] < g_logfile.cur_file_index) {
                for (int j = 0; j < cur_files_count; ++j) {
                    if (indexsort[i] == strtol(filessort[j] + (strlen(g_logfile.g_app_name) + strlen("_")), NULL, 10)) {
                        // 缝合目录和文件名形成完整文件路径
                        memset(filename, 0x00, sizeof(filename));
                        memcpy(filename, g_logfile.g_log_path, strlen(g_logfile.g_log_path));
                        memcpy(filename + strlen(g_logfile.g_log_path), filessort[j], strlen(filessort[j]));

                        // 删除旧的文件
                        remove(filename);
                        --removebefore;
                        --g_logfile.cur_files_count;

                    }
                }
            }

            if (removeafter > 0 && indexsort[i] > g_logfile.cur_file_index) {
                for (int j = 0; j < cur_files_count; ++j) {
                    if (indexsort[i] == strtol(filessort[j] + (strlen(g_logfile.g_app_name) + strlen("_")), NULL, 10)) {
                        // 缝合目录和文件名形成完整文件路径
                        memset(filename, 0x00, sizeof(filename));
                        memcpy(filename, g_logfile.g_log_path, strlen(g_logfile.g_log_path));
                        memcpy(filename + strlen(g_logfile.g_log_path), filessort[j], strlen(filessort[j]));

                        // 删除旧的文件
                        remove(filename);
                        --removeafter;
                        --g_logfile.cur_files_count;
                    }
                }
            }
        }
    }
    pthread_mutex_unlock(&g_logfile.cur_mutex);
}


void ne_someip_log_output(ne_someip_log_level_t level, const char* function, const char* file, int line, const char* fmt, ...)
{
    // 不需要log输出
    if (level >= NE_SOMEIP_LOG_LEVEL_NOLOG) {
        return;
    }

    // 比设定等级低的log不输出
    if (level < g_logfile.g_log_level) {
        return;
    }

    // 格式化输出格式问题
    if (NULL == fmt) {
        return;
    }

    char msg_buff[MAX_LOG_SIZE] = { 0 };
    va_list list;
    va_start(list, fmt);
    vsnprintf(msg_buff, sizeof(msg_buff), fmt, list);
    va_end(list);

    char tm_buff[32] = {0};
    struct timespec ts;
    clock_gettime( CLOCK_REALTIME, &ts );
    if ( ts.tv_nsec >= 1000000000LL ) {
        ts.tv_sec++;
        ts.tv_nsec -= 1000000000LL;
    }

    struct tm tmptime;
    const struct tm *const ptime = gmtime_r( &ts.tv_sec, &tmptime );

    if ( ptime == &tmptime ) {
        snprintf(tm_buff, sizeof(tm_buff), "%4d-%02d-%02d %02d:%02d:%02d.%ld",
                ptime->tm_year + 1900, ptime->tm_mon + 1, ptime->tm_mday,
                ptime->tm_hour, ptime->tm_min, ptime->tm_sec, ts.tv_nsec / 1000000);
    }

    char log_msg[MAX_LOG_SIZE] = { 0 };
    snprintf(log_msg, MAX_LOG_SIZE, NE_SOMEIP_LOG_FORMAT, tm_buff, log_level_str[level], g_logfile.g_app_name,
        getpid(), ne_someip_log_gettid(), function, NE_SOMEIP_FILE_NAME(file), line, msg_buff);

    ne_someip_log_write(log_msg, strlen(log_msg));

    if (g_logfile.g_log_console) {
        switch(level) {
            case NE_SOMEIP_LOG_LEVEL_WARNING:
                //黄色
                printf("\033[1;33m%s\033[0m", log_msg);
                break;
            case NE_SOMEIP_LOG_LEVEL_ERROR:
            case NE_SOMEIP_LOG_LEVEL_FATAL:
                //红色
                printf("\033[1;31m%s\033[0m", log_msg);
                break;
            case NE_SOMEIP_LOG_LEVEL_DEBUG:
            case NE_SOMEIP_LOG_LEVEL_INFO:
            default:
                printf("%s", log_msg);
                break;
        }
    }
}

void ne_someip_log_deinit()
{
    pthread_mutex_destroy(&g_logfile.cur_mutex);
    if (g_logfile.pfile != NULL) {
        fclose(g_logfile.pfile);
        g_logfile.pfile = NULL;
    }
}

/* EOF */
