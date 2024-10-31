#pragma once
#include <iostream>
#include <vector>

namespace hozon {
namespace netaos {
namespace mcap {

struct AttachmentHeader {
    int64_t header_length;
    std::string file_path;
};

std::string GetDirName(std::string path);

std::string GetFileName(std::string path);

}  // namespace mcap
}  // namespace netaos
}  // namespace hozon
