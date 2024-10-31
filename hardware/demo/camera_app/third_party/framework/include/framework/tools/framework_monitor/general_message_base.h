#ifndef TOOLS_CVT_MONITOR_GENERAL_MESSAGE_BASE_H_
#define TOOLS_CVT_MONITOR_GENERAL_MESSAGE_BASE_H_

#include <map>
#include <utility>

#include "framework/cyber.h"
#include "framework/tools/framework_monitor/renderable_message.h"

class Screen;

class GeneralMessageBase : public RenderableMessage {
 protected:
  static void PrintMessage(GeneralMessageBase* baseMsg,
                           const google::protobuf::Message& msg,
                           int* jump_lines, const Screen* s, int* line_no,
                           int indent);
  static void PrintField(GeneralMessageBase* baseMsg,
                         const google::protobuf::Message& msg, int* jump_lines,
                         const Screen* s, int* line_no, int indent,
                         const google::protobuf::Reflection* ref,
                         const google::protobuf::FieldDescriptor* field,
                         int index);

  static int LineCount(const google::protobuf::Message& msg, int screen_width);
  static int LineCountOfField(const google::protobuf::Message& msg,
                              int screen_width,
                              const google::protobuf::FieldDescriptor* field,
                              const google::protobuf::Reflection* reflection,
                              bool is_folded = true);

  void InsertRepeatedMessage(int line_no, GeneralMessageBase* item) {
    children_map_.emplace(line_no, item);
  }

  RenderableMessage* Child(int line_no) const override;

  explicit GeneralMessageBase(RenderableMessage* parent = nullptr)
      : RenderableMessage(parent), children_map_() {}
  ~GeneralMessageBase(void) { clear(); }

  void clear(void) {
    for (auto& iter : children_map_) {
      delete iter.second;
    }

    children_map_.clear();
  }

  GeneralMessageBase(const GeneralMessageBase&) = delete;
  GeneralMessageBase& operator=(const GeneralMessageBase&) = delete;

  std::map<const int, GeneralMessageBase*> children_map_;
};

#endif  // TOOLS_CVT_MONITOR_GENERAL_MESSAGE_BASE_H_
