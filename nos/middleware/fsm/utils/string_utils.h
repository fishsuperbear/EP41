#pragma once

#include <string>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace hozon {
namespace fsmcore {

bool safe_stoi(const std::string& str, int32_t& val);

bool safe_stoul(const std::string& str, uint32_t& val);

bool safe_stoull(const std::string& str, uint64_t& val);

bool safe_stoll(const std::string& str, int64_t& val);

double safe_stod(const std::string& str, double& val);

bool safe_stof(const std::string& str, float& val);

size_t string_to_char(const std::string& str, char* buff, const size_t buff_size);

void char_to_string(const char* buff, const size_t buff_size, std::string& str);

std::vector<std::string> split(const std::string& str, const std::string& pattern);

std::string trim(const std::string& s);

}
}
