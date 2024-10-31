#include "hw_impl_plat_log.h"

#include <string.h>
#include <stdio.h>

#define HALLOG_CTRL_LEVEL_MIN         1
#define HALLOG_CTRL_LEVEL_MAX         8

s32 hw_plat_logcontext_fill_bydefault(struct hw_plat_logcontext_t* o_pcontext)
{
    /*
    * There's no default log buffer setting operation here to enable logbuf bufmode.
    * So currently we can not set default bufmode to HW_PLAT_LOGCONTEXT_BUFMODE_LOGBUF.
    */
    STATIC_ASSERT(HW_PLAT_LOGCONTEXT_BUFMODE_DEFAULT != HW_PLAT_LOGCONTEXT_BUFMODE_LOGBUF);
    o_pcontext->bufmode = HW_PLAT_LOGCONTEXT_BUFMODE_DEFAULT;
    // do not need to set plogbuf (default is not bufmode)
    // do not need to set atomic_offset (default is not bufmode)
    // do not need to set totalbytecountmax (default is not bufmode)
    // do not need to set logbuflevel (default is not bufmode)
    o_pcontext->bytecountmax = HW_PLAT_LOGCONTEXT_BUFBYTECOUNT_MAX_DEFAULT;
    o_pcontext->level = HW_PLAT_LOGCONTEXT_LEVEL_DEFAULT;
    /*
    * There's no default log oper setting operation here to enable other opermode except
    * HW_PLAT_LOGOPER_MODE_INNER_IMPLEMENT.
    */
    STATIC_ASSERT(HW_PLAT_LOGOPER_MODE_DEFAULT == HW_PLAT_LOGOPER_MODE_INNER_IMPLEMENT);
    o_pcontext->logoper.opermode = HW_PLAT_LOGOPER_MODE_DEFAULT;
    strcpy(o_pcontext->logoper.innerimpl.logdir, HW_PLAT_LOGOPER_INNER_IMPLEMENT_FILE_LOGDIR_DEFAULT);
    strcpy(o_pcontext->logoper.innerimpl.logname, HW_PLAT_LOGOPER_INNER_IMPLEMENT_FILE_LOGNAME_DEFAULT);
    o_pcontext->logoper.innerimpl.bmktimedir = HW_PLAT_LOGOPER_BMKTIMEDIR_DEFAULT;
    o_pcontext->logoper.innerimpl.baddpidsuffix = HW_PLAT_LOGOPER_BADDPIDSUFFIX_DEFAULT;
    o_pcontext->logoper.innerimpl.syncmode = HW_PLAT_LOGOPER_FILE_SYNCMODE_DEFAULT;
    o_pcontext->logoper.innerimpl.runtime.binit = 0;
    // do not need to set filehandle (filehandle is invalid when binit is 0)
    strcpy(o_pcontext->logoper.innerimpl.runtime.logpath, "");
    return 0;
}

s32 hw_plat_logcontext_fill_bufmode_logbuf(struct hw_plat_logcontext_t* io_pcontext,
    char* i_plogbuf, u32 i_totalbytecountmax, u32 i_logbuflevel,
    struct hw_atomic_u32_t* i_patomic_offset, u32* i_poffset_initvalue)
{
    io_pcontext->bufmode = HW_PLAT_LOGCONTEXT_BUFMODE_LOGBUF;
    io_pcontext->plogbuf = i_plogbuf;
    io_pcontext->totalbytecountmax = i_totalbytecountmax;
    io_pcontext->logbuflevel = i_logbuflevel;
    if (!ISTWOPOWER(io_pcontext->totalbytecountmax)) {
        return -1;
    }
    if (io_pcontext->totalbytecountmax < 0x100000) {
        return -2;
    }
    if (io_pcontext->logbuflevel >= HW_LOG_LEVEL_UNMASK) {
        return -2;
    }
    io_pcontext->atomic_offset = i_patomic_offset;
    if (i_poffset_initvalue) {
        hw_plat_atomic_set_u32(io_pcontext->atomic_offset, *i_poffset_initvalue);
    }
    return 0;
}

s32 hw_plat_logcontext_init(struct hw_plat_logcontext_t* i_pcontext)
{
    if (HW_UNLIKELY(i_pcontext->logoper.opermode == HW_PLAT_LOGOPER_MODE_USER_DEFINED))
    {
        /*
        * Check user defined function is NOT NULL.
        */
        if (HW_UNLIKELY(i_pcontext->logoper.userdefined.pfunc_init == NULL))
        {
            hw_impl_plat_loginner_output("userdefined init is unexpected NULL!\r\n");
            hw_impl_plat_loginner_outputoff();
            return -1;
        }
        if (HW_UNLIKELY(i_pcontext->logoper.userdefined.pfunc_logoutput_valist == NULL))
        {
            hw_impl_plat_loginner_output("userdefined logoutput is unexpected NULL!\r\n");
            hw_impl_plat_loginner_outputoff();
            return -1;
        }
        if (HW_UNLIKELY(i_pcontext->logoper.userdefined.pfunc_flush == NULL))
        {
            hw_impl_plat_loginner_output("userdefined flush is unexpected NULL!\r\n");
            hw_impl_plat_loginner_outputoff();
            return -1;
        }
        if (HW_UNLIKELY(i_pcontext->logoper.userdefined.pfunc_deinit == NULL))
        {
            hw_impl_plat_loginner_output("userdefined deinit is unexpected NULL!\r\n");
            hw_impl_plat_loginner_outputoff();
            return -1;
        }
        return i_pcontext->logoper.userdefined.pfunc_init(i_pcontext);
    }
    if (HW_UNLIKELY(i_pcontext->logoper.opermode != HW_PLAT_LOGOPER_MODE_INNER_IMPLEMENT))
    {
        hw_impl_plat_loginner_output("opermode is NOT HW_PLAT_LOGOPER_MODE_INNER_IMPLEMENT, value[%u]!\r\n",
            i_pcontext->logoper.opermode);
        hw_impl_plat_loginner_outputoff();
        return -1;
    }
    if (i_pcontext->bufmode == HW_PLAT_LOGCONTEXT_BUFMODE_LOGBUF)
    {
        /*
        * Check 2 power totalbytecountmax when bufmode is logbuf.
        */
        if (!ISTWOPOWER(i_pcontext->totalbytecountmax)) {
            hw_impl_plat_loginner_output("totalbytecountmax is NOT 2~N when bufmode[LOGBUF], value[%x]!\r\n",
                i_pcontext->totalbytecountmax);
            hw_impl_plat_loginner_outputoff();
            return -1;
        }
        /*
        * Check totalbytecountmax is at least 0x100000 when bufmode is logbuf.
        */
        if (i_pcontext->totalbytecountmax < 0x100000) {
            hw_impl_plat_loginner_output("totalbytecountmax is smaller than 0x100000, value[%x]!\r\n",
                i_pcontext->totalbytecountmax);
            hw_impl_plat_loginner_outputoff();
            return -1;
        }
        /*
        * Check the logbuflevel can not be bigger than level.
        */
        if (i_pcontext->logbuflevel > i_pcontext->level) {
            hw_impl_plat_loginner_output("logbuflevel[%u] is bigger than level[%u]!\r\n",
                i_pcontext->logbuflevel, i_pcontext->level);
            hw_impl_plat_loginner_outputoff();
            return -1;
        }
        /*
        * Check the logbuf log level.
        */
        if (i_pcontext->logbuflevel >= HW_LOG_LEVEL_UNMASK) {
            hw_impl_plat_loginner_output("logbuflevel[%u] is not smaller than HW_LOG_LEVEL_UNMASK!\r\n",
                i_pcontext->logbuflevel);
            hw_impl_plat_loginner_outputoff();
            return -1;
        }
        /*
        * Check the log level.
        */
        if (i_pcontext->level >= HW_LOG_LEVEL_UNMASK) {
            hw_impl_plat_loginner_output("level[%u] is not smaller than HW_LOG_LEVEL_UNMASK!\r\n",
                i_pcontext->level);
            hw_impl_plat_loginner_outputoff();
            return -1;
        }
    }
    /*
    * HW_PLAT_LOGOPER_MODE_INNER_IMPLEMENT opermode.
    */
    /*
    * Currently, we only support HW_PLAT_LOGOPER_FILE_SYNCMODE_SINGLEPROCESS mode.
    */
    if (HW_UNLIKELY(i_pcontext->logoper.innerimpl.syncmode != HW_PLAT_LOGOPER_FILE_SYNCMODE_SINGLEPROCESS))
    {
        hw_impl_plat_loginner_output("syncmode is NOT HW_PLAT_LOGOPER_FILE_SYNCMODE_SINGLEPROCESS, value[%u]!\r\n",
            i_pcontext->logoper.innerimpl.syncmode);
        hw_impl_plat_loginner_outputoff();
        return -1;
    }
    /*
    * Calculate the directory to create.
    */
    char* logdir = &i_pcontext->logoper.innerimpl.runtime.logpath[0];
    char command[HW_PLAT_LOGOPER_INNER_IMPLEMENT_FILE_LOGPATHBYTE_MAX];
    char timestr[HW_PLAT_LOCALTIME_DESCBYTECOUNT_MAX];
    char pidstr[16];
    if (strcmp(i_pcontext->logoper.innerimpl.logdir, "") == 0)
    {
        strcpy(logdir, HW_PLAT_LOGOPER_INNER_IMPLEMENT_FILE_LOGDIR_DEFAULT);
    }
    else
    {
        strcpy(logdir, i_pcontext->logoper.innerimpl.logdir);
    }
    if (i_pcontext->logoper.innerimpl.bmktimedir == 1)
    {
        if (logdir[strlen(logdir) - 1] != '/')
        {
            strcat(logdir, "/");
        }
        hw_plat_localtime_getdescstr(timestr, NULL);
        strcat(logdir, timestr);
    }
    s32 exitstatus;
    strncpy(command, "mkdir -p ", sizeof(command));
    strncat(command, logdir, sizeof(command) - strlen(command) - 1);
    // snprintf(command, sizeof(command), "mkdir -p %s", logdir);
    hw_impl_plat_system(command, &exitstatus);
    /*
    * Calculate the log path to put log.
    */
    if (logdir[strlen(logdir) - 1] != '/')
    {
        strcat(logdir, "/");
    }
    if (strcmp(i_pcontext->logoper.innerimpl.logname, "") == 0)
    {
        strcat(logdir, HW_PLAT_LOGOPER_INNER_IMPLEMENT_FILE_LOGNAME_DEFAULT);
    }
    else
    {
        size_t logdir_len = strlen(logdir);
        size_t logname_len = strlen(i_pcontext->logoper.innerimpl.logname);
        size_t max_len = sizeof(logdir) - logdir_len - 1;
        size_t copy_len = (logname_len < max_len) ? logname_len : max_len;

        // Replace the use of strncat with memmove to avoid accessing overlapping memory regions
        // Copy the first n characters of src to dest, ensuring that the copy is safe even if the memory regions overlap
        memmove(logdir + logdir_len, i_pcontext->logoper.innerimpl.logname, copy_len);
        logdir[logdir_len + copy_len] = '\0';
    }
    /*
    * Add pid as suffix if needed.
    */
    if (i_pcontext->logoper.innerimpl.baddpidsuffix == 1)
    {
        sprintf(pidstr, "_%d", getpid());
        strcat(logdir, pidstr);
    }
    strcat(logdir, ".txt");
    /*
    * We will overwrite the old file if exist.
    */
    hw_impl_plat_file_open(&i_pcontext->logoper.innerimpl.runtime.filehandle, logdir);
    /*
    * Currently, only HW_PLAT_LOGOPER_FILE_SYNCMODE_SINGLEPROCESS mode.
    * We should init the offset value for cas add operation.
    * We store the offset atomic variable in the file handle.
    */
    hw_impl_plat_file_setcasaddmode(&i_pcontext->logoper.innerimpl.runtime.filehandle);
    hw_plat_atomic_set_u32(&i_pcontext->logoper.innerimpl.runtime.logindex, 0);
    if (i_pcontext->bufmode == HW_PLAT_LOGCONTEXT_BUFMODE_LOGBUF)
    {
        hw_plat_atomic_set_u32(&i_pcontext->logoper.innerimpl.runtime.logbuflogindex, 0);
        i_pcontext->boverflow = 0;
    }
    i_pcontext->logoper.innerimpl.runtime.binit = 1;
    return 0;
}

#define HW_IMPL_PLAT_LOGOUTPUT_VALIST_LOGHEAD_FORMAT        "[%s][%s][logindex:%s-%s][%s:%u]\t"

s32 hw_plat_logoutput_valist(struct hw_plat_logcontext_t* i_pcontext, u32 i_level, struct hw_plat_loghead_t* i_phead,
    struct hw_plat_logext_t* i_pext, const char* i_pformat, va_list i_valist)
{
    if (HW_UNLIKELY(i_pext != NULL))
    {
        if (HW_UNLIKELY(i_pext->pfunc_logoutput == NULL))
        {
            hw_impl_plat_loginner_output("pfunc_logoutput is unexpected NULL!\r\n");
            hw_impl_plat_loginner_outputoff();
            return -1;
        }
        return i_pext->pfunc_logoutput(i_pcontext, NULL, i_phead, i_pext, i_pformat, i_valist);
    }
    if (HW_UNLIKELY(i_pcontext->logoper.opermode == HW_PLAT_LOGOPER_MODE_USER_DEFINED))
    {
        return i_pcontext->logoper.userdefined.pfunc_logoutput_valist(i_pcontext, NULL, i_phead, i_pext, i_pformat, i_valist);
    }
    /*
    * First check whether has logoutput off.
    */
    u32 onstatus;
    hw_plat_logoutput_getonoffstatus(&onstatus);
    if (HW_UNLIKELY(onstatus == 0))
    {
        return -1;
    }
    char* plogbuf;
    s32 len = 0, headlen = 0, contentlen = 0;
    u32 logindex;
    char logindexstr[10];
    u32 logbuflogindex;
    char logbuflogindexstr[10];
    char* plogbufmalloc;
    const char* loglevelstr;
    const char* pfunc;
    u32 line;
    char ticktimestr[HW_PLAT_TICKTIME_DESCBYTECOUNT_MAX];
    hw_plat_ticktime_getdescstr(ticktimestr, NULL);
    switch (i_level)
    {
    case HW_LOG_LEVEL_DEBUG:
        loglevelstr = " debug";
        break;
    case HW_LOG_LEVEL_TRACE:
        loglevelstr = " trace";
        break;
    case HW_LOG_LEVEL_INFO:
        loglevelstr = "  info";
        break;
    case HW_LOG_LEVEL_WARN:
        loglevelstr = "  warn";
        break;
    case HW_LOG_LEVEL_ERR:
        loglevelstr = "   err";
        break;
    case HW_LOG_LEVEL_UNMASK:
        loglevelstr = "unmask";
        break;
    case HW_LOG_LEVEL_FATAL:
        loglevelstr = " fatal";
        break;
    default:
        loglevelstr = "levelNA";
        break;
    }
    if (i_level < i_pcontext->level)
    {
        strcpy(logindexstr, "NA");
    }
    else
    {
        hw_plat_atomic_cas_exchangeadd_u32(&i_pcontext->logoper.innerimpl.runtime.logindex,
            1, &logindex, NULL);
        sprintf(logindexstr, "0x%x", logindex);
    }
    if (i_phead != NULL)
    {
        if (i_phead->func == NULL)
        {
            pfunc = "NA";
        }
        else
        {
            pfunc = i_phead->func;
        }
        if (i_phead->func == NULL && i_phead->file == NULL)
        {
            line = 0;
        }
        else
        {
            line = i_phead->line;
        }
    }
    if (HW_LIKELY(i_pcontext->bufmode == HW_PLAT_LOGCONTEXT_BUFMODE_LOGBUF))
    {
        if (HW_UNLIKELY(i_level < i_pcontext->logbuflevel))
        {
            return -1;
        }
        hw_plat_atomic_cas_exchangeadd_u32(&i_pcontext->logoper.innerimpl.runtime.logbuflogindex,
            1, &logbuflogindex, NULL);
        sprintf(logbuflogindexstr, "0x%x", logbuflogindex);

        if (i_phead != NULL)
        {
            headlen = snprintf(NULL, 0, HW_IMPL_PLAT_LOGOUTPUT_VALIST_LOGHEAD_FORMAT, ticktimestr, loglevelstr, logindexstr, logbuflogindexstr, pfunc, line);
        }
        else
        {
            headlen = 0;
        }
        contentlen = vsnprintf(NULL, 0, i_pformat, i_valist);
        len = headlen + contentlen + 1;
        if (len > HW_PLAT_LOGCONTEXT_BUFBYTECOUNT_MAX_DEFAULT)
        {
            len = HW_PLAT_LOGCONTEXT_BUFBYTECOUNT_MAX_DEFAULT;
        }
        u32 startoffset, neweroffset;
        while (1)
        {
            hw_plat_atomic_cas_exchangeadd_u32(i_pcontext->atomic_offset, len, &startoffset, &neweroffset);
            if (HW_UNLIKELY(i_pcontext->boverflow == 0))
            {
                if (neweroffset >= i_pcontext->totalbytecountmax)
                {
                    i_pcontext->boverflow = 1;
                }
            }
            if ((startoffset & (i_pcontext->totalbytecountmax - 1))
                > (neweroffset & (i_pcontext->totalbytecountmax - 1)))
            {
                /*
                * When ringbuffer roll back, we cas add again.
                */
                continue;
            }
            startoffset = (startoffset & (i_pcontext->totalbytecountmax - 1));
            neweroffset = (neweroffset & (i_pcontext->totalbytecountmax - 1));
            /*
            * Write log head and content.
            */
            if (i_phead != NULL)
            {
                sprintf(i_pcontext->plogbuf + startoffset, HW_IMPL_PLAT_LOGOUTPUT_VALIST_LOGHEAD_FORMAT, ticktimestr, loglevelstr, logindexstr, logbuflogindexstr, pfunc, line);
            }
            vsnprintf(i_pcontext->plogbuf + startoffset + headlen, len - headlen, i_pformat, i_valist);
            break;
        }
        plogbuf = i_pcontext->plogbuf + startoffset;
        if (i_level < i_pcontext->level)
        {
            return -1;
        }
    }
    else if (HW_LIKELY(i_pcontext->bufmode == HW_PLAT_LOGCONTEXT_BUFMODE_STACK))
    {
        strcpy(logbuflogindexstr, "NA");
        if (i_level < i_pcontext->level)
        {
            return -1;
        }
        char stackbuf[HW_PLAT_LOGCONTEXT_STACKSIZE_MAX];
        char* pstackbuf = stackbuf;
        if (i_phead != NULL)
        {
            headlen = snprintf(pstackbuf, HW_PLAT_LOGCONTEXT_STACKSIZE_MAX, HW_IMPL_PLAT_LOGOUTPUT_VALIST_LOGHEAD_FORMAT, ticktimestr, loglevelstr, logindexstr, logbuflogindexstr, pfunc, line);
        }
        pstackbuf += headlen;
        contentlen = vsnprintf(pstackbuf, HW_PLAT_LOGCONTEXT_STACKSIZE_MAX - headlen, i_pformat, i_valist);
        plogbuf = stackbuf;
        len = headlen + contentlen;
    }
    else
    {
        strcpy(logbuflogindexstr, "NA");
        if (i_level < i_pcontext->level)
        {
            return -1;
        }
        /*
        * HW_PLAT_LOGCONTEXT_BUFMODE_DYNAMICMALLOC mode.
        */
        if (i_phead != NULL)
        {
            headlen = snprintf(NULL, 0, HW_IMPL_PLAT_LOGOUTPUT_VALIST_LOGHEAD_FORMAT, ticktimestr, loglevelstr, logindexstr, logbuflogindexstr, pfunc, line);
        }
        else
        {
            headlen = 0;
        }
        contentlen = vsnprintf(NULL, 0, i_pformat, i_valist);
        if (contentlen < 0)
        {
            hw_impl_plat_loginner_output("vsnprintf return <0 ret=%d!\r\n", contentlen);
            hw_impl_plat_loginner_outputoff();
            return -1;
        }
        len = headlen + contentlen + 1;
        if (len > HW_PLAT_LOGCONTEXT_BUFBYTECOUNT_MAX_DEFAULT)
        {
            len = HW_PLAT_LOGCONTEXT_BUFBYTECOUNT_MAX_DEFAULT;
        }
        plogbufmalloc = (char*)malloc(len);
        char* plogbufmalloccurr = plogbufmalloc;
        if (i_phead != NULL)
        {
            sprintf(plogbufmalloccurr, HW_IMPL_PLAT_LOGOUTPUT_VALIST_LOGHEAD_FORMAT, ticktimestr, loglevelstr, logindexstr, logbuflogindexstr, pfunc, line);
        }
        plogbufmalloccurr += headlen;
        vsnprintf(plogbufmalloccurr, len - headlen, i_pformat, i_valist);
        plogbuf = plogbufmalloc;
    }
    /*
    * Do log operation inner operation.
    */
    /*
    * Get the current file offset position to write to file.
    */
    hw_impl_plat_file* pimplfile = (hw_impl_plat_file*)&i_pcontext->logoper.innerimpl.runtime.filehandle;
    u32 offset;
    hw_plat_atomic_cas_exchangeadd_u32(&pimplfile->pcontrol->atomic_u32, len, &offset, NULL);
    FILE* fp = fopen(i_pcontext->logoper.innerimpl.runtime.logpath, "r+");
    fseek(fp, offset, SEEK_SET);
    fwrite(plogbuf, len, 1, fp);
    fclose(fp);
    if (HW_UNLIKELY(i_pcontext->bufmode == HW_PLAT_LOGCONTEXT_BUFMODE_DYNAMICMALLOC) && !plogbufmalloc )
    {
        free(plogbufmalloc);
    }
    if (i_level == HW_LOG_LEVEL_FATAL)
    {
        /*
        * First, output the backtrace.
        */
        hw_impl_plat_logdump_backtrace_without_checkoutputstage();
        if (HW_LIKELY(i_pcontext->bufmode == HW_PLAT_LOGCONTEXT_BUFMODE_LOGBUF))
        {
            /*
            * When logbuf mode, we need to dump all of the logbuf content out.
            * Currently, we dump it to a seperate file(logpath plus "fatal").
            */
            char logpath[HW_PLAT_LOGOPER_INNER_IMPLEMENT_FILE_LOGPATHBYTE_MAX + 10];
            strcpy(logpath, i_pcontext->logoper.innerimpl.runtime.logpath);
            strcat(logpath, "fatal");
            FILE* fpdump = fopen(logpath, "w");
            /*
            * First output the current offset to the end, and then from begin to the current offset.
            */
            u32 offsetdump;
            hw_plat_atomic_get_u32(i_pcontext->atomic_offset, &offsetdump);
            if (i_pcontext->boverflow == 1)
            {
                fwrite(&i_pcontext->plogbuf[offsetdump], i_pcontext->totalbytecountmax - offsetdump, 1, fpdump);
            }
            fwrite(&i_pcontext->plogbuf[0], offsetdump, 1, fpdump);
            fflush(fpdump);
            fclose(fpdump);
        }
        hw_impl_plat_loginner_outputoff();
        hw_plat_logoutput_flush(i_pcontext);
    }
    return 0;
}

s32 hw_plat_logoutput(struct hw_plat_logcontext_t* i_pcontext, u32 i_level, struct hw_plat_loghead_t* i_phead,
    struct hw_plat_logext_t* i_pext, const char* i_pformat, ...)
{
    va_list valist;
    va_start(valist, i_pformat);
    s32 ret = hw_plat_logoutput_valist(i_pcontext, i_level, i_phead, i_pext, i_pformat, valist);
    va_end(valist);
    return ret;
}

s32 hw_plat_logoutput_flush(struct hw_plat_logcontext_t* i_pcontext)
{
    if (HW_UNLIKELY(i_pcontext->logoper.opermode == HW_PLAT_LOGOPER_MODE_USER_DEFINED))
    {
        return i_pcontext->logoper.userdefined.pfunc_flush(i_pcontext);
    }
    FILE* fp;
    hw_impl_plat_file_getfilep(&i_pcontext->logoper.innerimpl.runtime.filehandle, &fp);
    fflush(fp);
    return 0;
}

s32 hw_plat_loglevel_set(struct hw_plat_logcontext_t* i_pcontext, u32 i_value)
{
    if ((i_value >= HALLOG_CTRL_LEVEL_MIN) && (i_value <= HALLOG_CTRL_LEVEL_MAX) && (i_value != i_pcontext->level)) {
        i_pcontext->level = i_value;
    } else {
        return -1;
    }

    return 0;
}
