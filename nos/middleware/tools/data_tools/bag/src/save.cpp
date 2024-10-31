#include "save.h"
#include "impl/save_impl.h"

namespace hozon {
namespace netaos {
namespace bag {

Save::Save() {
    save_impl_ = std::make_unique<SaveImpl>();
}

Save::~Save() {}

SaveErrorCode Save::Start(SaveOptions save_option) {
    return save_impl_->Start(save_option);
}

}  // namespace bag
}  //namespace netaos
}  //namespace hozon