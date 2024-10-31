#include "can_tsync_center/sig_stop.h"

namespace hozon {
namespace netaos {

bool SigHandler::_term_signal = false;
std::condition_variable SigHandler::_term_cv;
std::mutex SigHandler::_term_mutex;

}
}