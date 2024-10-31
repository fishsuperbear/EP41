#include "impl/attachment_impl.h"
#include "attachment.h"

namespace hozon {
namespace netaos {
namespace bag {

Attachment::Attachment() {
    attachment_impl_ = std::make_unique<AttachmentImpl>();
};

Attachment::~Attachment() {
    if (attachment_impl_) {
        attachment_impl_ = nullptr;
    }
};

AttachmentErrorCode Attachment::Start(AttachmentOptions attachment_option) {
    return attachment_impl_->Start(attachment_option);
};

}  // namespace bag
}  //namespace netaos
}  //namespace hozon
