#include "monitor/renderable_message.h"

#include <ncurses/ncurses.h>

#include "monitor/screen.h"

namespace hozon {
namespace netaos {
namespace topic {

void RenderableMessage::SplitPages(int key) {
    switch (key) {
        case CTRL('d'):
        case KEY_NPAGE:
            ++page_index_;
            if (page_index_ >= pages_) {
                page_index_ = pages_ - 1;
            }
            break;

        case CTRL('u'):
        case KEY_PPAGE:
            --page_index_;
            if (page_index_ < 1) {
                page_index_ = 0;
            }
            break;
        default: {
        }
    }
}
}  // namespace topic
}  // namespace netaos
}  // namespace hozon