#include "impl/attachment_impl.h"
#include <fstream>
#include <sys/stat.h>

namespace hozon {
namespace netaos {
namespace bag {

AttachmentImpl::AttachmentImpl() {
    reader_ = std::make_unique<ReaderImpl>();
    writer_ = std::make_unique<WriterImpl>();
    write_split_callback = std::bind(&AttachmentImpl::onWriterEvent, this, std::placeholders::_1);
    writer_->WriterRegisterCallback(write_split_callback);
};

AttachmentImpl::~AttachmentImpl() {
    
};

AttachmentErrorCode AttachmentImpl::Start(AttachmentOptions attachment_option) {
    force_write_ = attachment_option.force_write_;

    reader_->Open(attachment_option.url);
    WriterOptions writer_options;
    if (attachment_option.rewrite_name_.size())
        writer_options.output_file_name = attachment_option.rewrite_name_;
    else 
        writer_options.output_file_name = attachment_option.url.substr(0, attachment_option.url.find_last_of('.')) + "_new";
    writer_options.use_time_suffix = false;
    if (attachment_option.rewrite_attachment_) {
        rewrite_list_ = std::move(attachment_option.rewrite_list_);
        rewrite_attachment();
    } else if (attachment_option.add_new_attachment_) {
        add_list_ = std::move(attachment_option.add_list_);
        add_new_attachments();
    } else if (attachment_option.show_attachment_) {
        show_list_ = std::move(attachment_option.show_list_);
        show_attachments();
        reader_->Close();
        return AttachmentErrorCode::SUCCESS;
    } else if (attachment_option.extract_attachment_) {
        show_list_ = std::move(attachment_option.extract_list_);
        extract_attachments();
        reader_->Close();
        return AttachmentErrorCode::SUCCESS;
    } else {
        std::cout << "no match function" << std::endl;
    }

    writer_->Open(writer_options);

    auto topic_to_type = reader_->GetAllTopicsAndTypes();
    while (reader_->HasNext()) {
        auto msg = reader_->ReadNext();
        eprosima::fastrtps::rtps::SerializedPayload_t payload;
        payload.reserve(msg->serialized_data->buffer_length);
        memcpy(payload.data, msg->serialized_data->buffer, msg->serialized_data->buffer_length);
        payload.length = msg->serialized_data->buffer_length;
        writer_->write(payload, msg->topic_name, topic_to_type[msg->topic_name], msg->time_stamp);
    }

    reader_->Close();

    old_path_ = boost::filesystem::absolute(attachment_option.url);
    
    return AttachmentErrorCode::SUCCESS;
}

void AttachmentImpl::rewrite_attachment() {
    // printf("AttachmentImpl::rewrite_attachment()\n");
    std::string attachment_path;
    std::string attachment_name_in_bag;
    reader_->get_all_attachments_filepath(attachment_list_in_bag_);
    for (const auto& attachment_name: attachment_list_in_bag_) {
        auto attachment_ = reader_->read_attachment(attachment_name);
        if (attachment_ != nullptr) {
            rewrite_map_[attachment_name] = attachment_;
        } else {
            std::cout << "reader_->read_attachment() error : " << attachment_name << std::endl;
        }
    }
    for (const auto& str: rewrite_list_) {
        // printf("rewrite_list_ : %s\n", str.c_str());
        size_t pos = str.find(":");
        if (pos != std::string::npos) {
            attachment_path = str.substr(0, pos);
            attachment_name_in_bag = str.substr(pos + 1);
        } else {
            std::cout << "format does not meet the requirements : " << str << std::endl;
            continue;
        }
        if (std::find(attachment_list_in_bag_.begin(), attachment_list_in_bag_.end(), attachment_name_in_bag) != attachment_list_in_bag_.end()) {
            std::cout << attachment_name_in_bag << " is already exist in bag. if you want overwrite, please use -f" << std::endl;
            if (force_write_ == false)
                continue;
        }
        auto attachment_ = path_convert_to_attachment(attachment_path, attachment_name_in_bag);
        if (attachment_ != nullptr) {
            rewrite_map_[attachment_name_in_bag] = attachment_;
        } else {
            std::cout << "path_convert_to_attachment() error : " << attachment_path << std::endl;
        }
    }
    for (const auto& attachment : rewrite_map_) {
        // printf("rewrite_map_.first : %s\n", attachment.first.c_str());
        // printf("rewrite_map_.second : %s\n", attachment.second->name.c_str());
        writer_->add_attachment_to_list(attachment.second);
    }
}

void AttachmentImpl::add_new_attachments() {
    // printf("AttachmentImpl::add_new_attachments()\n");
    reader_->get_all_attachments_filepath(attachment_list_in_bag_);
    for (const auto& attachment_name: attachment_list_in_bag_) {
        auto attachment_ = reader_->read_attachment(attachment_name);
        if (attachment_ != nullptr) {
            // printf("add old attachment : %s\n", attachment_name.c_str());
            writer_->add_attachment_to_list(attachment_);
        } else {
            std::cout << "reader_->read_attachment() error : " << attachment_name << std::endl;
        }
    }
    for (const auto& attachment_name: add_list_) {
        auto attachment_ = path_convert_to_attachment(attachment_name, "");
        if (attachment_ != nullptr) {
            writer_->add_attachment_to_list(attachment_);
        } else {
            std::cout << "path_convert_to_attachment() error : " << attachment_name << std::endl;
        }
    }
}

void AttachmentImpl::show_attachments() {
    for (const auto& attachment_name: show_list_) {
        auto attachment_ = reader_->read_attachment(attachment_name);
        if (attachment_ != nullptr) {
            std::cout << attachment_->name << " : " << std::endl;
            std::cout << attachment_->data << std::endl << std::endl << std::endl;
        } else {
            std::cout << "reader_->read_attachment() error : " << attachment_name << std::endl;
        }
    }
}

std::shared_ptr<rosbag2_storage::Attachment> AttachmentImpl::path_convert_to_attachment(std::string attachment_path, std::string rename) {
    std::ifstream ifs(attachment_path, std::ifstream::in);
    if (!ifs.good()) {
        std::cout << attachment_path << " not exist." << std::endl;
        return nullptr;
    }
    struct stat s;
    if (stat(attachment_path.c_str(), &s) == 0 && S_ISDIR(s.st_mode)) {
        std::cout << attachment_path << " is a dir not file." << std::endl;
        return nullptr;
    }
    boost::filesystem::path absolutePath;
    if (attachment_path[0] != '/')
        absolutePath = boost::filesystem::current_path() / boost::filesystem::canonical(attachment_path);
    else
        absolutePath = boost::filesystem::canonical(attachment_path);
    auto attachment_ = std::make_shared<rosbag2_storage::Attachment>();
    attachment_->logTime = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    attachment_->createTime = 0;
    if (rename.size())
        attachment_->name = rename;
    else 
        attachment_->name = absolutePath.string();
    attachment_->mediaType = "normal_file";
    std::cout << "add attachment : "<< attachment_->name << std::endl;
    if (ifs.is_open()) {
        int length = 0;
        ifs.seekg(0, std::ios::end);
        length = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        char* buffer = new char[length];
        ifs.read(buffer, length);
        ifs.close();
        attachment_->data.assign(buffer, length);
        attachment_->dataSize = length;
        delete[] buffer;
    } else {
        return nullptr;
    }
    ifs.close();
    return attachment_;
}

void AttachmentImpl::extract_attachments() {
    // 创建空文件
    std::string attachment_path_output;
    std::string attachment_name_in_bag;
    boost::filesystem::path open_name;
    for (const auto& attachment_name: show_list_) {
        size_t pos = attachment_name.find(":");
        if (pos != std::string::npos) {
            attachment_name_in_bag = attachment_name.substr(0, pos);
            attachment_path_output = attachment_name.substr(pos + 1);
            open_name = attachment_path_output;
            if (boost::filesystem::is_directory(open_name)) {
                open_name = open_name.string() + ((open_name.string()[open_name.size() - 1] == '/') ? "" : "/") + boost::filesystem::path(attachment_name_in_bag).filename().string();
            }
        } else {
            attachment_name_in_bag = attachment_name;
            open_name = attachment_name_in_bag.substr(0, pos);
            open_name = open_name.filename();
        }
        std::ofstream fileStream(open_name.string());
        fileStream.close();
        // 再次打开文件流并写入内容
        std::ofstream appendStream(open_name.string(), std::ios::app);
        if (appendStream.is_open()) {
            auto attachment_ = reader_->read_attachment(attachment_name_in_bag);
            if (attachment_ != nullptr) {
                appendStream << attachment_->data;
                appendStream.close();
                std::cout << "file write success : " <<  open_name.string() << std::endl;
            } else {
                std::cout << "reader_->read_attachment() error : " << attachment_name_in_bag << std::endl;
            }
        } else {
            std::cout << "file write error : " <<  open_name.string() << std::endl;
        }
    }
}

void AttachmentImpl::onWriterEvent(hozon::netaos::bag::WriterInfo& info) {
    if (info.state == InfoType::FILE_CLOSE) {
        // new_path_ = boost::filesystem::absolute(info.file_path);
        
        // printf("oldPath  : %s\n", old_path_.string().c_str());
        // printf("newPath  : %s\n", new_path_.string().c_str());

        std::cout << "new bag: " << info.file_path << std::endl;
        // boost::system::error_code ec;
        // boost::filesystem::rename(new_path_, old_path_, ec);
        // if (ec) {
        //     std::cerr << "Error renaming file: " << ec.message() << std::endl;
        // }
    }
}

}  // namespace bag
}  //namespace netaos
}  //namespace hozon
