#ifndef TOOLS_CVT_MONITOR_GENERAL_MESSAGE_H_
#define TOOLS_CVT_MONITOR_GENERAL_MESSAGE_H_

#include "monitor/general_message_base.h"
#include "google/protobuf/descriptor.pb.h"

namespace hozon {
namespace netaos {
namespace topic {

class Screen;

class GeneralMessage : public GeneralMessageBase {
   public:
    GeneralMessage(GeneralMessageBase* parent, const google::protobuf::Message* msg, const google::protobuf::Reflection* reflection, const google::protobuf::FieldDescriptor* field);

    ~GeneralMessage() {
        field_ = nullptr;
        message_ptr_ = nullptr;
        reflection_ptr_ = nullptr;
    }

    int Render(const Screen* s, int key) override;

   private:
    GeneralMessage(const GeneralMessage&) = delete;
    GeneralMessage& operator=(const GeneralMessage&) = delete;

    int item_index_;
    bool is_folded_;
    const google::protobuf::FieldDescriptor* field_;
    const google::protobuf::Message* message_ptr_;
    const google::protobuf::Reflection* reflection_ptr_;
};

}  // namespace topic
}  // namespace netaos
}  // namespace hozon

#endif  // TOOLS_CVT_MONITOR_GENERAL_MESSAGE_H_
