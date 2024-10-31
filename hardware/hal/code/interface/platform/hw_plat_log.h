#ifndef HW_PLAT_LOG_H
#define HW_PLAT_LOG_H

#include "hw_plat_basic.h"

/*
* We do NOT define log level here!
* Because other platform code like hw_plat_xxx.h may use the log
* level defines.
*/

__BEGIN_DECLS

enum HW_PLAT_LOGCONTEXT_BUFMODE
{
	HW_PLAT_LOGCONTEXT_BUFMODE_MIN = 0,
	HW_PLAT_LOGCONTEXT_BUFMODE_MINMINUSONE = HW_PLAT_LOGCONTEXT_BUFMODE_MIN - 1,

	/*
	* It is recommended to use the mode to enhance performance.
	*/
	HW_PLAT_LOGCONTEXT_BUFMODE_LOGBUF,
	/*
	* The system will use HW_PLAT_LOGCONTEXT_BUFMODE_DYNAMICMALLOC mode instead of
	* HW_PLAT_LOGCONTEXT_BUFMODE_STACK when do block log output operation.
	*/
	HW_PLAT_LOGCONTEXT_BUFMODE_STACK,
	HW_PLAT_LOGCONTEXT_BUFMODE_DYNAMICMALLOC,

	HW_PLAT_LOGCONTEXT_BUFMODE_MAXADDONE,
	HW_PLAT_LOGCONTEXT_BUFMODE_MAX = HW_PLAT_LOGCONTEXT_BUFMODE_MAXADDONE - 1,
};

#define HW_PLAT_LOGCONTEXT_BUFMODE_DEFAULT			HW_PLAT_LOGCONTEXT_BUFMODE_STACK

/*
* Valid when the platform log context is HW_PLAT_LOGCONTEXT_BUFMODE_STACK mode.
* The program will static assign a array of HW_PLAT_LOGCONTEXT_STACKSIZE_MAX in the stack.
* In future, may use variable-length array in GNU compiler.
* It will NOT affect block log output.
*/
#define HW_PLAT_LOGCONTEXT_STACKSIZE_MAX					512
#define HW_PLAT_LOGCONTEXT_BUFBYTECOUNT_MAX_DEFAULT			1024
/*
* 64M
*/
#define HW_PLAT_LOGCONTEXT_TOTALBUFBYTECOUNT_MAX_DEFAULT	0x10000

/*
* Log level of logbuf.
* Any log which level is equal to or bigger than the value will be output.
*/
#define HW_PLAT_LOGCONTEXT_LOGBUFLEVEL_DEFAULT				HW_LOG_LEVEL_DEBUG
/*
* Log level of log output.
*/
#define HW_PLAT_LOGCONTEXT_LEVEL_DEFAULT					HW_LOG_LEVEL_INFO

enum HW_PLAT_LOGOPER_MODE
{
	HW_PLAT_LOGOPER_MODE_MIN = 0,
	HW_PLAT_LOGOPER_MODE_MINMINUSONE = HW_PLAT_LOGOPER_MODE_MIN - 1,

	/*
	* Currently, the inner implement is to output to file.
	* The file can be set or use the default file.
	* You can set hw_plat_logoperinnerimpl_t when use the mode.
	*/
	HW_PLAT_LOGOPER_MODE_INNER_IMPLEMENT,
	/*
	* Call user defined function like
	* pfunc_init/pfunc_logoutput_valist/pfunc_flush/pfunc_deinit
	* when call logoutput functions or
	* hw_plat_logcontext_init/hw_plat_logcontext_flush/hw_plat_logcontext_deinit.
	*/
	HW_PLAT_LOGOPER_MODE_USER_DEFINED,

	HW_PLAT_LOGOPER_MODE_MAXADDONE,
	HW_PLAT_LOGOPER_MODE_MAX = HW_PLAT_LOGOPER_MODE_MAXADDONE - 1,
};

#define HW_PLAT_LOGOPER_MODE_DEFAULT							HW_PLAT_LOGOPER_MODE_INNER_IMPLEMENT

#define HW_PLAT_LOGOPER_INNER_IMPLEMENT_FILE_LOGDIRBYTE_MAX		192
#define HW_PLAT_LOGOPER_INNER_IMPLEMENT_FILE_LOGNAMEBYTE_MAX	64

/*
* It is the default directory to put hal log.
*/
#define HW_PLAT_LOGOPER_INNER_IMPLEMENT_FILE_LOGDIR_DEFAULT		"./hallog"
/*
* It is the default name to put hal log in the directory.
* The system will add the suffix .txt to the end of the name.
*/
#define HW_PLAT_LOGOPER_INNER_IMPLEMENT_FILE_LOGNAME_DEFAULT	"hallog"

enum HW_PLAT_LOGOPER_FILE_SYNCMODE
{
	HW_PLAT_LOGOPER_FILE_SYNCMODE_MIN = 0,
	HW_PLAT_LOGOPER_FILE_SYNCMODE_MINMINUSONE = HW_PLAT_LOGOPER_FILE_SYNCMODE_MIN - 1,

	/*
	* Just use append mode to output log to the file without any lock.
	*/
	HW_PLAT_LOGOPER_FILE_SYNCMODE_APPEND_WITHOUT_LOCK,
	/*
	* This mode consider the situation that more than one thread in the process may output log.
	*/
	HW_PLAT_LOGOPER_FILE_SYNCMODE_SINGLEPROCESS,
	/*
	* It is for debug only, for it is performance limited.
	*/
	HW_PLAT_LOGOPER_FILE_SYNCMODE_FILELOCK,

	HW_PLAT_LOGOPER_FILE_SYNCMODE_MAXADDONE,
	HW_PLAT_LOGOPER_FILE_SYNCMODE_MAX = HW_PLAT_LOGOPER_FILE_SYNCMODE_MAXADDONE - 1,
};

#define HW_PLAT_LOGOPER_BMKTIMEDIR_DEFAULT						1
#define HW_PLAT_LOGOPER_BADDPIDSUFFIX_DEFAULT					1
#define HW_PLAT_LOGOPER_FILE_SYNCMODE_DEFAULT					HW_PLAT_LOGOPER_FILE_SYNCMODE_SINGLEPROCESS

#define HW_PLAT_LOGOPER_INNER_IMPLEMENT_FILE_LOGPATHBYTE_MAX	256

typedef struct hw_plat_logoperinnerimplruntime_t
{
	/*
	* 1 means has init and can log output.
	* 0 means has not init and cannot log output.
	* Will be set to 0 when calling hw_plat_logcontext_fill_bydefault function.
	* Will be set to 1 after successfully calling hw_plat_logcontext_init function.
	*/
	u32									binit;
	/*
	* Valid when binit is 1.
	* Invalid when binit is 0.
	* The file handle is created by hw_plat_logcontext_init function when opermode is
	* HW_PLAT_LOGOPER_MODE_INNER_IMPLEMENT.
	*/
	struct hw_plat_filehandle_t			filehandle;
	/*
	* log index plus one every log record in logbuf.
	*/
	struct hw_atomic_u32_t				logbuflogindex;
	/*
	* log index plus one every log output.
	*/
	struct hw_atomic_u32_t				logindex;
	char								logpath[HW_PLAT_LOGOPER_INNER_IMPLEMENT_FILE_LOGPATHBYTE_MAX];
} hw_plat_logoperinnerimplruntime_t;

typedef struct hw_plat_logoperinnerimpl_t
{
	/*
	* You can set logdir to "" to mean use HW_PLAT_LOGOPER_INNER_IMPLEMENT_FILE_LOGDIR_DEFAULT.
	*/
	char								logdir[HW_PLAT_LOGOPER_INNER_IMPLEMENT_FILE_LOGDIRBYTE_MAX];
	/*
	* You can set logdir to "" to mean use HW_PLAT_LOGOPER_INNER_IMPLEMENT_FILE_LOGNAME_DEFAULT.
	*/
	char								logname[HW_PLAT_LOGOPER_INNER_IMPLEMENT_FILE_LOGNAMEBYTE_MAX];
	/*
	* 1 means make a directory of the current time in the logdir.
	* 0 means do not make a directory of the current time in the logdir.
	* The default value is HW_PLAT_LOGOPER_BMKTIMEDIR_DEFAULT.
	*/
	u32									bmktimedir;
	/*
	* 1 means add pid as suffix. Use getpid() to get in linux.
	* 0 means not add any suffix.
	* The default value is HW_PLAT_LOGOPER_BADDPIDSUFFIX_DEFAULT.
	*/
	u32									baddpidsuffix;
	/*
	* The default value is HW_PLAT_LOGOPER_FILE_SYNCMODE_DEFAULT.
	*/
	HW_PLAT_LOGOPER_FILE_SYNCMODE		syncmode;
	/*
	* It will be init by hw_plat_logcontext_init.
	*/
	struct hw_plat_logoperinnerimplruntime_t	runtime;
} hw_plat_logoperinnerimpl_t;

typedef s32 (*hw_plat_logoperuserdefined_pfunc_init)(struct hw_plat_logcontext_t* i_pcontext);
/*
* Use the same func pointer whichever block mode or not.
* When not block mode, io_plogblock is NULL.
* Currently, the func pointer type is the same as hw_plat_logext_pfunc_logoutput.
*/
typedef s32(*hw_plat_logoperuserdefined_pfunc_logoutput_valist)(struct hw_plat_logcontext_t* i_pcontext, struct hw_plat_logblock_t* io_plogblock,
	struct hw_plat_loghead_t* i_phead, struct hw_plat_logext_t* i_pext, const char* i_pformat, va_list i_valist);
typedef s32 (*hw_plat_logoperuserdefined_pfunc_flush)(struct hw_plat_logcontext_t* i_pcontext);
typedef s32 (*hw_plat_logoperuserdefined_pfunc_deinit)(struct hw_plat_logcontext_t* i_pcontext);

typedef struct hw_plat_logoperuserdefined_t
{
	hw_plat_logoperuserdefined_pfunc_init				pfunc_init;
	hw_plat_logoperuserdefined_pfunc_logoutput_valist	pfunc_logoutput_valist;
	hw_plat_logoperuserdefined_pfunc_flush				pfunc_flush;
	hw_plat_logoperuserdefined_pfunc_deinit				pfunc_deinit;
} hw_plat_logoperuserdefined_t;

/*
* We use the structure to define how to use the format output buffer to log output.
*/
typedef struct hw_plat_logoper_t
{
	/*
	* The default value is HW_PLAT_LOGOPER_MODE_DEFAULT.
	* The opermode choose affect the following two members.
	*/
	HW_PLAT_LOGOPER_MODE				opermode;
	/*
	* Valid when opermode is HW_PLAT_LOGOPER_MODE_INNER_IMPLEMENT.
	*/
	struct hw_plat_logoperinnerimpl_t	innerimpl;
	/*
	* Valid when opermode is HW_PLAT_LOGOPER_MODE_USER_DEFINED.
	*/
	struct hw_plat_logoperuserdefined_t	userdefined;
} hw_plat_logoper_t;

typedef struct hw_plat_logcontext_t
{
	/*
	* The default value is HW_PLAT_LOGCONTEXT_BUFMODE_DEFAULT.
	* The bufmode choose affect the following three members.
	*/
	HW_PLAT_LOGCONTEXT_BUFMODE			bufmode;
	/*
	* It is valid only when bufmode is HW_PLAT_LOGCONTEXT_BUFMODE_LOGBUF.
	* It is the buffer start pointer.
	* When mode is HW_PLAT_LOGCONTEXT_BUFMODE_LOGBUF, it can NOT be NULL.
	* When other bufmode, the system will ignore the member.
	*/
	char*								plogbuf;
	/*
	* It is valid only when bufmode is HW_PLAT_LOGCONTEXT_BUFMODE_LOGBUF.
	* It is the pointer to the atomic offset u32 value.
	*/
	struct hw_atomic_u32_t*				atomic_offset;
	/*
	* It is valid only when bufmode is HW_PLAT_LOGCONTEXT_BUFMODE_LOGBUF.
	* The total byte count max of the plogbuf.
	* The value should be 2 power number.
	* You can set it to HW_PLAT_LOGCONTEXT_TOTALBUFBYTECOUNT_MAX_DEFAULT which is a
	* reference value.
	*/
	u32									totalbytecountmax;
	/*
	* 1 means has overflow the full logbuf.
	* 0 means has not overflow the full logbuf.
	*/
	u32									boverflow;
	/*
	* It is valid only when bufmode is HW_PLAT_LOGCONTEXT_BUFMODE_LOGBUF.
	* When logbuf mode, we use the level to control what log any be stored on the logbuf.
	* The default value is HW_PLAT_LOGCONTEXT_LOGBUFLEVEL_DEFAULT.
	* When do context init, the logbuflevel can not be bigger than level.
	* The value should always be smaller than HW_LOG_LEVEL_UNMASK.
	*/
	u32									logbuflevel;

	/*
	* Byte count max of one log.
	* Not affect block output mode.(block mode use reservedbytecount to manage)
	* Max log string byte count of one output log function call(include the log head).
	* It will affect every bufmode mode except HW_PLAT_LOGCONTEXT_BUFMODE_STACK mode.
	* When is HW_PLAT_LOGCONTEXT_BUFMODE_STACK mode, the stack size is constant value
	* that is HW_PLAT_LOGCONTEXT_STACKSIZE_MAX.
	*/
	u32									bytecountmax;

	/*
	* The default value is HW_PLAT_LOGCONTEXT_LEVEL_DEFAULT.
	* The value should always be smaller than HW_LOG_LEVEL_UNMASK.
	*/
	u32									level;

	/*
	* We currently only implement log by file of the inner implement.
	* We can set the user defined implement.
	*/
	struct hw_plat_logoper_t			logoper;
} hw_plat_logcontext_t;

/*
* func and file can be NULL to mean no details about it.
*/
typedef struct hw_plat_loghead_t
{
	const char*							func;
	const char*							file;
	// not valid when func and file are all NULL
	u32									line;
} hw_plat_loghead_t;

/*
* Please make sure it is NOT too big to put into stack.
*/
#define HW_PLAT_LOGBLOCK_RESERVED_BYTECOUNT_DEFAULT			4096
#define HW_PLAT_LOGBLOCK_OUTPUTIMMEDIATE_DEFAULT			0

/*
* We use block to reserve a place to put continuous log string so that we can read it more conveniently
* later. You should not change it by yourself during the block log output period.
*/
typedef struct hw_plat_logblock_t
{
	/*
	* The default value is HW_PLAT_LOGBLOCK_RESERVED_BYTECOUNT_DEFAULT.
	*/
	u32									reservedbytecount;
	/*
	* The default value is HW_PLAT_LOGBLOCK_OUTPUTIMMEDIATE_DEFAULT.
	* 1 means output immediately.
	* 0 means output when call hw_plat_logblock_finish.
	*/
	u32									boutputimmediate;
	/*
	* It will copy the logcontext content to here when do hw_plat_logblock_init_bydefault function.
	*/
	struct hw_plat_logcontext_t			logcontext;
} hw_plat_logblock_t;

#define HW_PLAT_LOGEXT_NUMPVOIDS_MAX		4

/*
* Use the same func pointer whichever block mode or not.
* When not block mode, io_plogblock is NULL.
*/
typedef s32 (*hw_plat_logext_pfunc_logoutput)(struct hw_plat_logcontext_t* i_pcontext, struct hw_plat_logblock_t* io_plogblock,
	struct hw_plat_loghead_t* i_phead, struct hw_plat_logext_t* i_pext, const char* i_pformat, va_list i_valist);

typedef struct hw_plat_logext_t
{
	void*								parray_pvoid[HW_PLAT_LOGEXT_NUMPVOIDS_MAX];
	/*
	* Use the same func pointer whichever block mode or not.
	* When not block mode, io_plogblock is NULL.
	*/
	hw_plat_logext_pfunc_logoutput		pfunc_logoutput;
} hw_plat_logext_t;

#define HW_PLAT_LOGOUTPUT_INNER_PATH_DEFAULT		"./innerlog_hw_platform.txt"
#define HW_PLAT_LOGOUTPUT_INNER_PATH_BYTECOUNT_MAX	256

/*
* The default inner log path is HW_PLAT_LOGOUTPUT_INNER_PATH_DEFAULT.
* The log path byte count should be less than HW_PLAT_LOGOUTPUT_INNER_PATH_BYTECOUNT_MAX.
* return 0 means change success.
* return <0 means change fail.
*/
s32 hw_plat_innerlog_path_change(char* i_path);
/*
* Always return 0.
* When o_ponstatus is not NULL, *o_ponstatus is the log onoff status.
* *o_ponstatus is 1 means still output log
* *o_ponstatus is 0 means output log off.
*/
s32 hw_plat_logoutput_getonoffstatus(u32* o_ponstatus);
/*
* Always return 0.
* Mainly used by user defined log function.
* Currently, we can only change on to off, but cannot change off back to on.
* When you use inner implemented logoutput level FATAL, it will trigger the status to off.
* When you use inner implemented logoutput function, you do not need to use the current api.
*/
s32 hw_plat_logoutput_setoffstatus();

/*
* See the note of hw_plat_logcontext_t.
*/
s32 hw_plat_logcontext_fill_bydefault(struct hw_plat_logcontext_t* o_pcontext);
/*
* Set the bufmode to logbuf and set the logbuf pointer and init offset by input parameters.
* You can set *i_poffset_initvalue to the init value you want to set to the i_patomic_offset.
* You can set i_poffset_initvalue to NULL to mean you let i_patomic_offset the origin value.
* The input i_totalbytecountmax should be 2~N. And i_totalbytecountmax should be at least
* 0x100000.
* return -1 means the input i_totalbytecountmax is not 2~N.
* return -2 means the input i_totalbytecountmax is smaller than 0x100000.
* return -3 means the input i_logbuflevel is not smaller than HW_LOG_LEVEL_UNMASK.
* return 0 means success.
*/
s32 hw_plat_logcontext_fill_bufmode_logbuf(struct hw_plat_logcontext_t* io_pcontext,
	char* i_plogbuf, u32 i_totalbytecountmax, u32 i_logbuflevel,
	struct hw_atomic_u32_t* i_patomic_offset, u32* i_poffset_initvalue);
/*
* You should call the function before calling any logoutput function.
* The function will output the init log.
*/
s32 hw_plat_logcontext_init(struct hw_plat_logcontext_t* i_pcontext);
s32 hw_plat_logcontext_flush(struct hw_plat_logcontext_t* i_pcontext);
/*
* The function will call hw_plat_logcontext_flush first and then close the
*/
s32 hw_plat_logcontext_deinit(struct hw_plat_logcontext_t* i_pcontext);

/*
* i_phead and i_pext can be NULL.
* When i_phead is NULL, it will not output log head.
* When i_pext is not NULL, the logoutput function pointer of i_pext should not be NULL.
*/
s32 hw_plat_logoutput_valist(struct hw_plat_logcontext_t* i_pcontext, u32 i_level, struct hw_plat_loghead_t* i_phead,
	struct hw_plat_logext_t* i_pext, const char* i_pformat, va_list i_valist);
/*
* i_phead and i_pext can be NULL.
* When i_phead is NULL, it will not output log head.
*/
s32 hw_plat_logoutput(struct hw_plat_logcontext_t* i_pcontext, u32 i_level, struct hw_plat_loghead_t* i_phead,
	struct hw_plat_logext_t* i_pext, const char* i_pformat, ...);

/*
* Use default value to fill *io_plogblock.
* See note in hw_plat_logblock_t define for default value.
*/
s32 hw_plat_logblock_fill_bydefault(struct hw_plat_logcontext_t* i_pcontext, struct hw_plat_logblock_t* io_plogblock);
/*
* i_phead and i_pext can be NULL.
* When i_phead is NULL, it will not output log head.
*/
s32 hw_plat_logblockoutput_valist(struct hw_plat_logblock_t* io_plogblock, struct hw_plat_loghead_t* i_phead,
	struct hw_plat_logext_t* i_pext, const char* i_pformat, va_list i_valist);
/*
* i_phead and i_pext can be NULL.
* When i_phead is NULL, it will not output log head.
*/
s32 hw_plat_logblockoutput(struct hw_plat_logblock_t* io_plogblock, struct hw_plat_loghead_t* i_phead,
	struct hw_plat_logext_t* i_pext, const char* i_pformat, ...);
/*
* The function will flush output the block log when boutputimmediate is 0.
* The function will also free the malloc memory of the log block if has.
*/
s32 hw_plat_logblock_finish(struct hw_plat_logblock_t* io_plogblock);

s32 hw_plat_logoutput_flush(struct hw_plat_logcontext_t* i_pcontext);

s32 hw_plat_loglevel_set(struct hw_plat_logcontext_t* i_pcontext, u32 i_value);
__END_DECLS

#endif
