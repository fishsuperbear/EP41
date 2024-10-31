/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: system monitor info def
 */

#ifndef SYSTEM_MONITOR_INFO_DEF_H
#define SYSTEM_MONITOR_INFO_DEF_H

#include <stdint.h>
#include <vector>
#include <string>
#include <unordered_map>

#define CPU_CMD "/app/bin/mpstat -P ALL | awk '{print $3,$4,$5,$6,$7,$8,$9,$10,$11,$12}' | sed -n '4,$p'"
#define PID_CPU_CMD "/app/bin/pidstat -u |sort -r -n -k 8 | head -10 | awk '{print $2,$3,$4,$5,$6,$7,$8,$9,$10}'"
#define DISK_CMD "df -h | sed -n '2,$p'"
#define MEM_CMD "free -m | awk '{print $2,$3,$4,$5,$7}' | sed -n '2,$p'"
#define PID_MEM_CMD "/app/bin/pidstat -r |sort -r -n -k 8 | head -10 | awk '{print $2,$3,$4,$5,$6,$7,$8,$9}'"

const std::string CPU_INFO_TITLE = "CPU INFO:\n";
const std::string MEM_INFO_TITLE = "MEM INFO:\n";
const std::string DISK_INFO_TITLE = "DISK INFO:\n";
const std::string FILE_INFO_TITLE = "FILE INFO:\n";
const std::string MNAND_INFO_TITLE = "MNAND INFO:\n";
const std::string NETWORK_INFO_TITLE = "NETWORK INFO:\n";
const std::string PROCESS_INFO_TITLE = "PROCESS INFO:\n";
const std::string TEMP_INFO_TITLE = "TEMP INFO:\n";
const std::string VOLTAGE_INFO_TITLE = "VOLTAGE INFO:\n";
const std::string CONTROL_OUTPUT_LINE = "---------------------------------------------------------------------------------------------------\n";

const std::vector<std::string> MONITOR_INFO_ORDER = {
    "mem", "temp", "cpu", "voltage", "process", "disk", "network", "mnand", "file"
};

const std::vector<std::string> SYSTEM_MONITOR_INFO_ORDER = {
    "cpu", "mem", "disk"
};

const std::vector<std::string> DEVICE_MONITOR_INFO_ORDER = {
    "temp", "voltage", "mnand", "network"
};

const std::vector<std::string> SAFETY_MONITOR_INFO_ORDER = {
    "process", "file"
};

const std::unordered_map<std::string, std::vector<std::string>> MONITOR_RECORD_FILE_PATH_MAP = {
    {"cpu", {"CPU MONITOR INFO:\n", "/opt/usr/log/system_monitor_log/cpu_monitor.log", "/opt/usr/log/system_monitor_log/cpu_monitor.log.backup"}},
    {"mem", {"MEM MONITOR INFO:\n", "/opt/usr/log/system_monitor_log/mem_monitor.log", "/opt/usr/log/system_monitor_log/mem_monitor.log.backup"}},
    {"disk", {"DISK MONITOR INFO:\n", "/opt/usr/log/system_monitor_log/disk_monitor.log", "/opt/usr/log/system_monitor_log/disk_monitor.log.backup"}},
    {"file", {"FILE MONITOR INFO:\n", "/opt/usr/log/system_monitor_log/file_monitor.log", "/opt/usr/log/system_monitor_log/file_monitor.log.backup"}},
    {"mnand", {"MNAND MONITOR INFO:\n", "/opt/usr/log/system_monitor_log/mnand_monitor.log", "/opt/usr/log/system_monitor_log/mnand_monitor.log.backup"}},
    {"network", {"NETWORK MONITOR INFO:\n", "/opt/usr/log/system_monitor_log/network_monitor.log", "/opt/usr/log/system_monitor_log/network_monitor.log.backup"}},
    {"process", {"PROCESS MONITOR INFO:\n", "/opt/usr/log/system_monitor_log/process_monitor.log", "/opt/usr/log/system_monitor_log/process_monitor.log.backup"}},
    {"temp", {"TEMP MONITOR INFO:\n", "/opt/usr/log/system_monitor_log/temp_monitor.log", "/opt/usr/log/system_monitor_log/temp_monitor.log.backup"}},
    {"voltage", {"VOLTAGE MONITOR INFO:\n", "/opt/usr/log/system_monitor_log/voltage_monitor.log", "/opt/usr/log/system_monitor_log/voltage_monitor.log.backup"}}
};

struct CpuInfo {
    float usr;
    float nice;
    float sys;
    float iowait;
    float irq;
    float soft;
    float steal;
    float guest;
    float gnice;
    float idle;
};

struct PidCpuInfo {
    uint32_t uid;
    uint32_t pid;
    float usr;
    float system;
    float guest;
    float wait;
    float cpu;
    uint32_t cpuName;
    char command[100];
};

struct CpuStatus {
    CpuInfo cpu_all;
    std::vector<CpuInfo> cup_details;
};

struct DiskInfo {
    char fileSyetem[100];
    char size[10];
    char used[10];
    char avail[10];
    char use[10];
    char mountedOn[100];
};

struct MemInfo {
    uint32_t total;
    uint32_t used;
    uint32_t free;
    uint32_t shared;
    uint32_t available;
};

struct PidMemInfo {
    uint32_t uid;
    uint32_t pid;
    float minflt;
    float majflt;
    uint32_t vsz;
    uint32_t rss;
    float mem;
    char command[100];
};

struct TempInfo {
    float core;
    float local0;
    float remote0;
    float local1;
    float remote1;
};

#endif  // SYSTEM_MONITOR_INFO_DEF_H
