
#include "global_Info.h"


namespace state_machine {
int GlobalMsgSet::Init() {
        GlobalMsgSet::Instance().MsgSetInitial();
        GlobalMsgSet::Instance().ClearMsgOut();
        return 0;
}
}