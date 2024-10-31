#include "neta_node.h"

neta_log        __neta_log = {
    /*
    * We will frequently check the group count to check whether the 
    * log block group has inited or not.
    */
    .groupcount = 0,
    .parraypgroup = { NULL },
    .parrayppagegroup = { NULL },
    .parraypgroupshare = { NULL },
    .parrayppagegroupshare = { NULL },
    .expectedgroupindex = 0,
    .writerprocesscount = 0,
    .readerthreadcount = 0,
    .expectedreaderthreadid = 0,
    .btagdirty = 0,
    .bquit = 0,
    .alreadyhalfplogblockarray = { NULL },
    .breadyarray = { 0 },
    .wpalreadyhalf = 0,
    .rpalreadyhalf = 0,
    .alreadyhalfcounter = 0,
    .bclearinggroup = 0,
    .grouptoclearindex = INTERNAL_STRATEGY_NEEDTHREADPUTBLOCK_CLEAR_GROUP_INDEX_BEGIN,
};
