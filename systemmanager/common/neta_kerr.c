#include "neta_kerr.h"

const char* neta_kerr_info(s32 i_s32)
{
    switch (i_s32)
    {
    case NETA_KERR_COMMON_UNEXPECTED:
        return "kerr_common_unexpected";
    case NETA_KERR_COMMON_RETURN_NULL_PTR:
        return "kerr_common_return_null_ptr";
    case NETA_KERR_LOGBLOCKMAXCOUNT_PERBUDDY_POWER_CHECK:
        return "kerr_logblockmaxcount_perbuddy_power_check";
    case NETA_KERR_LOGBLOCKBUDDYMAXCOUNT_PERGROUP_POWER_CHECK:
        return "kerr_logblockbuddymaxcount_pergroup_power_check";
    case NETA_KERR_LOGBLOCKGROUP_NOTINIT:
        return "kerr_logblockgroup_notinit";
    case NETA_KERR_LOGBLOCKGROUP_BQUIT_IS_1:
        return "kerr_logblockgroup_bquit_is_1";
    case NETA_KERR_LOGBLOCKGROUP_NEEDNEW_INDEX_WRONG:
        return "kerr_logblockgroup_neednew_index_wrong";
    case NETA_KERR_LOGBLOCKGROUP_EXCEEDMAXCOUNT:
        return "kerr_logblockgroup_exceedmaxcount";
    case NETA_KERR_LOGBLOCKBUDDY_BRESERVED_1_NOT_SUPPORT:
        return "kerr_loblockbuddy_reserved_1_not_support";
    case NETA_KERR_LOGBLOCKUSER_ROLECHECKFAIL_CALLPRODUCERFUNC_WHENCONSUMER:
        return "kerr_logblockuser_rolecheckfail_callproducerfunc_whenconsumer";
    case NETA_KERR_LOGBLOCKUSER_ROLECHECKFAIL_CALLCONSUMERFUNC_WHENPRODUCER:
        return "kerr_logblockuser_rolecheckfail_callconsumerfunc_whenproducer";
    case NETA_KERR_LOGBLOCKUSER_ROLECHECKFAIL_CALLFINISHLOGBLOCK:
        return "kerr_logblockuser_rolecheckfail_callfinishlogblock";
    case NETA_KERR_LOGBLOCKUSER_THREADCOOKIE_ALREADY_NOT_0:
        return "kerr_logblockuser_threadcookie_already_not_0";
    case NETA_KERR_LOGBLOCKUSER_THREADCOOKIE_IS_0:
        return "kerr_logblockuser_threadcookie_is_0";
    case NETA_KERR_LOGBLOCKUSER_LOGBLOCK_DMABUFEXPORT_FAIL:
        return "kerr_logblockuser_logblock_dmabufexport_fail";
    case NETA_KERR_LOGBLOCKUSER_LOGBLOCKGROUPSHARE_DMABUFEXPORT_FAIL:
        return "kerr_logblockuser_logblockgroupshare_dmabufexport_fail";
    case NETA_KERR_LOGBLOCKUSER_LOGBLOCKGROUP_DMABUFEXPORT_FAIL:
        return "NETA_KERR_LOGBLOCKUSER_LOGBLOCKGROUP_DMABUFEXPORT_FAIL";
    case NETA_KERR_LOGBLOCKUSER_TRYPUT_CHECKREFCOUNT_NOT_2:
        return "kerr_logblockuser_tryput_checkrefcount_not_2";
    case NETA_KERR_LOGBLOCKUSER_CONSUMERCOUNT_CHECK_FAIL:
        return "kerr_logblockuser_consumercount_check_fail";
    case NETA_KERR_LOGBLOCKUSER_LOGBLOCKCOOKIE_IS_0:
        return "kerr_logblockuser_logblockcookie_is_0";
    default:
        return "kerr_unknown_code";
    }
}
