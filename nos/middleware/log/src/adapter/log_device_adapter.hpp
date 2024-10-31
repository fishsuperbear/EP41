/************************************************************************************
*  Copyright (c) Hozon Technologies Co., Ltd. 2022-2030. All rights reserved.       *
*                                                                                   *
*  @file     hz_log_adapter.hpp                                                     *
*  @brief    Define of class ILogDeviceAdapter                                          *
*  Details.                                                                         *
*                                                                                   *
*  @version  0.0.0.1                                                                *
*                                                                                   *
*-----------------------------------------------------------------------------------*
*  Change History :                                                                 *
*  <Date>     | <Version> | <Author>       | <Description>                          *
*-----------------------------------------------------------------------------------*
*  2022/06/15 | 0.0.0.1   | YangPeng      | Create file                             *
*-----------------------------------------------------------------------------------*
*                                                                                   *
*************************************************************************************/

#ifndef __LOG_DEVICE_ADAPTER_HPP__
#define __LOG_DEVICE_ADAPTER_HPP__

#include <iostream>
#include <memory>

namespace hozon {
namespace netaos {
namespace log {

/**
* @brief ILogDeviceAdapter class
* Interface class for operate the log device.
*/
class ILogDeviceAdapter
{
    public:
        /** 
        * @brief Constructor function of class ILogDeviceAdapter
        *
        */
        ILogDeviceAdapter(){}

        /** 
        * @brief Destructor function of class ILogDeviceAdapter
        * 
        */
        virtual ~ILogDeviceAdapter(){}

        /** 
        * @brief Init device function
        * 
        *
        */
        virtual void initDevice(bool rawLogFormat) = 0;

        /** 
        * @brief setLog2Terminal function
        * 
        * @param logto         true or false
        *
        */
        virtual void setLog2Terminal(bool logto) = 0;

        /** 
        * @brief setLog2File function
        * 
        * @param logto         true or false
        * @param filePath         file path
        * @param fileBaseName         file base name
        * @param maxLogFileNum        max Log File Num
        * @param maxSizeOfLogFile        max Size Of Log File
        *
        */
        virtual void setLog2File(bool logto, std::string filePath, std::string fileBaseName, std::uint32_t maxLogFileNum, std::uint64_t maxSizeOfLogFile) = 0;

        /** 
        * @brief setLog2Remote function
        * 
        * @param logto         true or false
        * @param serverIp         server Ip
        * @param port        port
        *
        */
        virtual void setLog2Remote(bool logto, std::string serverIp, std::uint32_t port) = 0;

       /** 
        * @brief setLog2LogService function
        * 
        * @param logto         true or false
        * @param fileBaseName         file base name
        *
        */
        virtual void setLog2LogService(bool logTo, std::string fileBaseName) = 0;

        /** 
        * @brief critical c++ stype function
        *       ---this function should be implemented in a subclass
        * 
        * @param message The message to be output
        *
        */
        virtual void critical(const std::string& message) = 0;


        /** 
        * @brief error c++ stype function
        *       ---this function should be implemented in a subclass
        * 
        * @param message The message to be output
        *
        */
        virtual void error(const std::string& message) = 0;


        /** 
        * @brief warn c++ stype function
        *       ---this function should be implemented in a subclass
        * 
        * @param message The message to be output
        *
        */
        virtual void warn(const std::string& message) = 0;


        /** 
        * @brief info c++ stype function
        *       ---this function should be implemented in a subclass
        * 
        * @param message The message to be output
        *
        */
        virtual void info(const std::string& message) = 0;

        /** 
        * @brief debug c++ stype function
        *       ---this function should be implemented in a subclass
        * 
        * @param message The message to be output
        *
        */
        virtual void debug(const std::string& message) = 0;

        /** 
        * @brief trace c++ stype function
        *       ---this function should be implemented in a subclass
        * 
        * @param message The message to be output
        *
        */
        virtual void trace(const std::string& message) = 0;

        virtual void setFileName(const std::string& name) = 0;

        virtual void setTerminalName(const std::string& name) = 0;

    private:

};




}
}
}
#endif  // __LOG_DEVICE_ADAPTER_HPP__ 
