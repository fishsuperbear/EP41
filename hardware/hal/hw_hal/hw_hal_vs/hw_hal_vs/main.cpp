#if 0

#include "hw_platform.h"

struct hw_plat_logcontext_t		_logcontext;

char _logringbuffer[0x2000000];
struct hw_atomic_u32_t	_offset;

#define HW_LOG_INFO(...)		do \
	{	\
		struct hw_plat_loghead_t _____internal_head;	\
		_____internal_head.func = __FUNCTION__;	\
		_____internal_head.file = __FILE__;	\
		_____internal_head.line = __LINE__;	\
		hw_plat_logoutput(&_logcontext, HW_LOG_LEVEL_INFO, &_____internal_head, NULL, __VA_ARGS__);	\
	} while(0)

#define HW_LOG_NOHEAD_INFO(...)		do \
	{	\
		hw_plat_logoutput(&_logcontext, HW_LOG_LEVEL_INFO, NULL, NULL, __VA_ARGS__);	\
	} while(0)

#define HW_LOG_FATAL(...)		do \
	{	\
		struct hw_plat_loghead_t _____internal_head;	\
		_____internal_head.func = __FUNCTION__;	\
		_____internal_head.file = __FILE__;	\
		_____internal_head.line = __LINE__;	\
		hw_plat_logoutput(&_logcontext, HW_LOG_LEVEL_FATAL, &_____internal_head, NULL, __VA_ARGS__);	\
	} while (0)

int main()
{
	s32 ret;
	ret = hw_plat_logcontext_fill_bydefault(&_logcontext);
	if (ret < 0) {
		printf("ret = %d\n", ret);
	}
	u32 initvalue = 0;
	ret = hw_plat_logcontext_fill_bufmode_logbuf(&_logcontext, 
		_logringbuffer, 0x2000000, HW_PLAT_LOGCONTEXT_LOGBUFLEVEL_DEFAULT,
		&_offset, &initvalue);
	if (ret < 0) {
		printf("ret = %d\n", ret);
	}
	ret = hw_plat_logcontext_init(&_logcontext);
	if (ret < 0) {
		printf("ret = %d\n", ret);
	}

	HW_LOG_INFO("info message: %d\r\n", 100);

	HW_LOG_INFO("info message: %d\r\n", 101);

	HW_LOG_INFO("info ");
	HW_LOG_NOHEAD_INFO("message ");
	HW_LOG_NOHEAD_INFO("1 ");
	HW_LOG_NOHEAD_INFO("2 ");
	HW_LOG_NOHEAD_INFO("3\r\n");

	HW_LOG_FATAL("fatal message: %d\r\n", 102);

	return 0;
}

#endif
