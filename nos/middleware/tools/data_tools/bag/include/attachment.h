#pragma once
#include <string>
#include <memory>
#include <vector>
#include <utility>
namespace hozon {
namespace netaos {
namespace bag {

struct AttachmentOptions {
    std::string url = "";
    std::vector<std::string> rewrite_list_; // <attachment path>:<attachments name in bag>
    std::string rewrite_name_ = "";
    std::vector<std::string> add_list_;     // <attachment path>
    std::vector<std::string> show_list_;    // <attachment path>
    std::vector<std::string> extract_list_;    // <attachment path>
    bool rewrite_attachment_ = false;
    bool add_new_attachment_ = false;
    bool show_attachment_ = false;
    bool force_write_ = false;
    bool extract_attachment_ = false;
};

enum class AttachmentErrorCode {
    SUCCESS = 0,
    FILE_FAILED = 1,
    READ_FILE_FAILED = 2,          //打开文件失败
    INVALID_FRAME = 3,             //无效帧
    TAKE_NEXT_MESSAGE_FAILED = 4,  //读取下一帧数据失败
    TO_JSON_FAILED = 5,            //proto转json失败
    FAILED_IDL_TYPE_SUPPORT = 6    //数据类型不支持
};

class AttachmentImpl;

class Attachment final {
   public:
    explicit Attachment();
    ~Attachment();

    AttachmentErrorCode Start(AttachmentOptions attachment_option);

   private:
    std::unique_ptr<AttachmentImpl> attachment_impl_;
};

}  // namespace bag
}  //namespace netaos
}  //namespace hozon