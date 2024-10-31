#pragma once
#include "reader_impl.h"
#include "writer_impl.h"
#include "attachment.h"
#include <map>
#include <boost/filesystem.hpp>

namespace hozon {
namespace netaos {
namespace bag {

class AttachmentImpl final {
   public:
    explicit AttachmentImpl();
    ~AttachmentImpl();
    AttachmentErrorCode Start(AttachmentOptions attachment_option);

   private:
    void onWriterEvent(hozon::netaos::bag::WriterInfo& info);
    std::shared_ptr<rosbag2_storage::Attachment> path_convert_to_attachment(std::string attachment_path, std::string rename);
    void rewrite_attachment();
    void add_new_attachments();
    void show_attachments();
    void extract_attachments();

   private:
    bool force_write_ = false;
    std::unique_ptr<ReaderImpl> reader_;
    std::unique_ptr<WriterImpl> writer_;

    std::vector<std::string> rewrite_list_;
    std::vector<std::string> add_list_;
    std::vector<std::string> show_list_;
    std::vector<std::string> extract_list_;

    std::vector<std::string> attachment_list_in_bag_;
    std::map<std::string, std::shared_ptr<rosbag2_storage::Attachment>> rewrite_map_;
    std::function<void(hozon::netaos::bag::WriterInfo&)> write_split_callback;

    boost::filesystem::path old_path_;
    boost::filesystem::path new_path_;
};

}  // namespace bag
}  //namespace netaos
}  //namespace hozon
