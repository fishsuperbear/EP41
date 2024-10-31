#include "list.h"
#include "impl/list_impl.h"

namespace hozon {
namespace netaos {
namespace topic {

List::List() {
    list_impl_ = std::make_unique<ListImpl>();
}

List::~List() {
    if (list_impl_) {
        list_impl_ = nullptr;
    }
}

void List::Start(ListOptions list_options) {
    list_impl_->Start(list_options);
}

}  // namespace topic
}  //namespace netaos
}  //namespace hozon
