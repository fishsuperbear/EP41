// Copyright 2018, Bosch Software Innovations GmbH.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ROSBAG2_STORAGE__LOGGING_HPP_
#define ROSBAG2_STORAGE__LOGGING_HPP_

#include <sstream>
#include <string>

#include "data_tools_logger.hpp"

using namespace hozon::netaos::data_tool_common;

#define ROSBAG2_STORAGE_PACKAGE_NAME " [rosbag2_storage] "

// #define ROSBAG2_STORAGE_LOG_CRITICAL_WITH_HEAD       BAG_LOG_CRITICAL_WITH_HEAD
// #define ROSBAG2_STORAGE_LOG_ERROR_WITH_HEAD          BAG_LOG_ERROR_WITH_HEAD
// #define ROSBAG2_STORAGE_LOG_WARN_WITH_HEAD           BAG_LOG_WARN_WITH_HEAD
// #define ROSBAG2_STORAGE_LOG_INFO_WITH_HEAD           BAG_LOG_INFO_WITH_HEAD
// #define ROSBAG2_STORAGE_LOG_DEBUG_WITH_HEAD          BAG_LOG_DEBUG_WITH_HEAD
// #define ROSBAG2_STORAGE_LOG_TRACE_WITH_HEAD          BAG_LOG_TRACE_WITH_HEAD

#define ROSBAG2_STORAGE_LOG_CRITICAL         COMMON_LOG_CRITICAL << ROSBAG2_STORAGE_PACKAGE_NAME
#define ROSBAG2_STORAGE_LOG_ERROR            COMMON_LOG_ERROR << ROSBAG2_STORAGE_PACKAGE_NAME
#define ROSBAG2_STORAGE_LOG_WARN             COMMON_LOG_WARN << ROSBAG2_STORAGE_PACKAGE_NAME
#define ROSBAG2_STORAGE_LOG_INFO             COMMON_LOG_INFO << ROSBAG2_STORAGE_PACKAGE_NAME
#define ROSBAG2_STORAGE_LOG_DEBUG            COMMON_LOG_DEBUG << ROSBAG2_STORAGE_PACKAGE_NAME
#define ROSBAG2_STORAGE_LOG_TRACE            COMMON_LOG_TRACE << ROSBAG2_STORAGE_PACKAGE_NAME

#endif  // ROSBAG2_STORAGE__LOGGING_HPP_
