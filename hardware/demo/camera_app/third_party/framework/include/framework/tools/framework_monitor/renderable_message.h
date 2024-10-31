#ifndef TOOLS_CVT_MONITOR_RENDERABLE_MESSAGE_H_
#define TOOLS_CVT_MONITOR_RENDERABLE_MESSAGE_H_

#include <string>

class Screen;

class RenderableMessage {
 public:
  static constexpr int FrameRatio_Precision = 2;

  explicit RenderableMessage(RenderableMessage* parent = nullptr,
                             int line_no = 0)
      : line_no_(line_no),
        pages_(1),
        page_index_(0),
        page_item_count_(24),
        parent_(parent),
        frame_ratio_(0.0) {}

  virtual ~RenderableMessage() { parent_ = nullptr; }

  virtual int Render(const Screen* s, int key) = 0;
  virtual RenderableMessage* Child(int /* line_no */) const = 0;

  virtual double frame_ratio(void) { return frame_ratio_; }

  RenderableMessage* parent(void) const { return parent_; }
  void set_parent(RenderableMessage* parent) {
    if (parent == parent_) {
      return;
    }
    parent_ = parent;
  }

  int page_item_count(void) const { return page_item_count_; }

 protected:
  int* line_no(void) { return &line_no_; }
  void set_line_no(int line_no) { line_no_ = line_no; }
  void reset_line_page(void) {
    line_no_ = 0;
    page_index_ = 0;
  }
  void SplitPages(int key);

  int line_no_;
  int pages_;
  int page_index_;
  int page_item_count_;
  RenderableMessage* parent_;
  double frame_ratio_;

  friend class Screen;
};  // RenderableMessage

#endif  // TOOLS_CVT_MONITOR_RENDERABLE_MESSAGE_H_
