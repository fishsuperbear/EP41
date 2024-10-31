// Copyright 2017 Open Source Robotics Foundation, Inc.
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

#include "ament_index_cpp/get_package_prefix.hpp"

#include <sys/stat.h>
#include <stdexcept>
#include <string>

// #include "ament_index_cpp/get_resource.hpp"
#include "ament_index_cpp/get_search_paths.hpp"

namespace ament_index_cpp {

static std::string format_package_not_found_error_message(const std::string& package_name) {
    std::string message = "package '" + package_name + "' not found, searching: [";
    auto search_paths = get_search_paths();
    for (const auto& path : search_paths) {
        message += path + ", ";
    }
    if (search_paths.size() > 0) {
        message = message.substr(0, message.size() - 2);
    }
    return message + "]";
}

PackageNotFoundError::PackageNotFoundError(const std::string& _package_name) : std::out_of_range(format_package_not_found_error_message(_package_name)), package_name(_package_name) {}

PackageNotFoundError::~PackageNotFoundError() {}

std::string get_package_prefix(const std::string& package_name) {
    //get the aAMENT_PREFIX_PATH
    auto prefix_path = get_search_paths().front();
    struct stat f_state;
    if (stat(prefix_path.c_str(), &f_state) == 0 && (f_state.st_mode & S_IFDIR) != 0) {
        return prefix_path;
    } else {
        throw PackageNotFoundError(package_name);
    }
}

}  // namespace ament_index_cpp
