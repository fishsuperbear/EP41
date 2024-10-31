#include <fstream>
#include <sstream>
#include <algorithm>
#include "split.h"
#include "rosbag2_cpp/reader.hpp"
#include "rcpputils/filesystem_helper.hpp"
#include "mcap_common.hpp"

namespace hozon {
namespace netaos {
namespace mcap {

Split::Split() {}

Split::~Split() {}

void Split::extract_attachments(std::vector<std::string> mcap_file_path_vec,
                                std::vector<std::string> attachment_file_path_vec,
                                std::string output_folder_path) {
    if (output_folder_path == "") {
        output_folder_path = "./";
    } else {
        rcpputils::fs::path db_path(output_folder_path);
        if (!db_path.is_directory()) {
            bool dir_created = rcpputils::fs::create_directories(db_path);
            if (!dir_created) {
                std::stringstream error;
                error << "Failed to create database directory (" << db_path.string() << ").";
                throw std::runtime_error{error.str()};
            }
        }
    }
    
    for (auto mcap_file_path : mcap_file_path_vec) {
        std::unique_ptr<rosbag2_cpp::Reader> reader;
        reader = std::make_unique<rosbag2_cpp::Reader>();
        reader->open(mcap_file_path);
        std::vector<std::string> tmp_attachment_file_path_vec;
        if (attachment_file_path_vec.empty()) {
            reader->get_all_attachments_filepath(tmp_attachment_file_path_vec);
        } else {
            tmp_attachment_file_path_vec = attachment_file_path_vec;
        }
        for (std::string attachment : tmp_attachment_file_path_vec) {
            std::shared_ptr<rosbag2_storage::Attachment> attachment_ptr = reader->read_Attachment(attachment);
            if (attachment_ptr != nullptr) {
                std::string attachment_output_folder_path = output_folder_path + '/' + GetDirName(attachment_ptr->name);
                rcpputils::fs::path db_path(attachment_output_folder_path);
                if (!db_path.is_directory()) {
                    bool dir_created = rcpputils::fs::create_directories(db_path);
                    if (!dir_created) {
                        std::stringstream error;
                        error << "Failed to create database directory (" << db_path.string() << ").";
                        reader->close();
                        throw std::runtime_error{error.str()};
                    }
                }
                std::string attachment_output_file_path = attachment_output_folder_path + '/' + GetFileName(attachment_ptr->name);
                std::ofstream attachment_output_file_of(attachment_output_file_path, std::ios::out | std::ios::binary);
                attachment_output_file_of << attachment_ptr->data;
                attachment_output_file_of.close();
                m_output_file_path_vec.push_back(attachment_output_file_path);
            }
        }
        reader->close();
    }
}

void Split::extract_attachments(std::vector<std::string> mcap_file_path_vec,
                                std::string output_folder_path) {
    std::vector<std::string> attachment_file_path_vec;
    extract_attachments(mcap_file_path_vec, attachment_file_path_vec, output_folder_path);
}

std::vector<std::string> Split::get_output_file_path_vec() {
    return m_output_file_path_vec;
}

}  // namespace mcap
}  // namespace netaos
}  // namespace hozon
