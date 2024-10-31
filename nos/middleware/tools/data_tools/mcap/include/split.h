#pragma once
#include <iostream>
#include <vector>

namespace hozon {
namespace netaos {
namespace mcap {

class Split {
public:
    Split();
    ~Split();
    void extract_attachments(std::vector<std::string> mcap_file_path_vec,
                             std::vector<std::string> attachment_file_path_vec,
                             std::string output_folder_path);
    void extract_attachments(std::vector<std::string> mcap_file_path_vec,
                             std::string output_folder_path);
    std::vector<std::string> get_output_file_path_vec();
        
private:
    std::vector<std::string> m_output_file_path_vec;
};

}  // namespace mcap
}  // namespace netaos
}  // namespace hozon
