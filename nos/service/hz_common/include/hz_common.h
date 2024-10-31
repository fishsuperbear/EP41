#pragma once

#include <cstdint>
#include <iostream>
#include <memory>

namespace hozon {
namespace netaos{
namespace common {

class PlatformCommonImpl;

class PlatformCommon {
   public:
    PlatformCommon();
    ~PlatformCommon();

/**
* @brief Provide initialization interface.
*
* @param [in] log_app_name : App name.
* @param [in] log_level : Log level #{0，1，2，3，4，5, 6}， 0：verbose, 1：debug, 2：info, 3：warn, 4：error, 5：fatal, 6：off.
* @param [in] log_mode : Log mode #{0, 1, 2, 3, 4, 5, 6}, 0: 固定文件存储，异步日志 1: 打屏，同步日志，适合调试， 2: 指定文件存储，同步日志，适合调试，
            #3：固定+打屏， 4：固定+指定， 5：打屏+指定， 6：固定+打屏+指定.
*
* @return PlatformCommon initializes the interface, if it returns non-zero, then it means there is an error.
*
* @attention This interface must be called first.
*/
    static int32_t Init(const std::string log_app_name, const uint32_t log_level, const uint32_t log_mode);

/**
* @brief Provide an interface to calculate the data interval of two frames.
*
* @param [in] topic_name : The topic name represents the name of the log frame that needs to be printed.
* @param [out] last_time_ms : The time of the last published data(ms).
*
* @return If it returns non-zero, then it means there is an error.
*
* @attention Nothing.
*/
	static int32_t CheckTwoFrameInterval(const std::string topic_name, uint64_t& last_time_ms);

/**
* @brief Provide an interface to get date time.
*
* @param [out] now_s : The time of the date time(s in utc format).
* @param [out] now_ns : The time of the date time(ns in utc format).
*
* @return If it returns non-zero, then it means there is an error.
*
* @attention Nothing.
*/
	static int32_t GetDataTime(uint32_t &now_s, uint32_t &now_ns);
    
/**
* @brief Provide an interface to get date time.
*
* @param [out] now_s : The time of the manage time(s in utc format).
* @param [out] now_ns : The time of the manage time(ns in utc format).
*
* @return If it returns non-zero, then it means there is an error.
*
* @attention Nothing.
*/
    static int32_t GetManageTime(uint32_t &now_s, uint32_t &now_ns);

   private:
    static std::unique_ptr<PlatformCommonImpl> pimpl_;
};

}  // namespace common
}  // namespace hozon
}
