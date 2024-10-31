#ifndef HW_PLAT_BASEDEF_H
#define HW_PLAT_BASEDEF_H

#include "hw_porting_baseinc.h"

#define HW_PLAT_CHIP_ARCH_ARM64		0
#define HW_PLAT_CHIP_ARCH_X64		1
#define HW_PLAT_CHIP_ARCH_X86		2


#if (HW_PLAT_PROJ == HW_PLAT_PROJ_ORIN)

#ifndef s8
typedef signed char			s8;
#endif
#ifndef s16
typedef short				s16;
#endif
#ifndef s32
typedef int					s32;
#endif
#ifndef s64
typedef long long			s64;
#endif
#ifndef u8
typedef unsigned char		u8;
#endif
#ifndef u16
typedef unsigned short		u16;
#endif
#ifndef u32
typedef unsigned int		u32;
#endif
#ifndef u64
typedef unsigned long long	u64;
#endif

#define HW_PLAT_CHIP_ARCH		HW_PLAT_CHIP_ARCH_ARM64
#define HW_PLAT_POINTER_BITNUM		64
#ifndef STATIC_ASSERT
#define STATIC_ASSERT(truecond)		static_assert (truecond, "error")
#endif

/*
* Normally, the ERG size is cache line size. ERG(Exclusive Reservation Granule).
*/
#define HW_PLAT_ERG_BYTECOUNT		32

#endif


#ifndef __BEGIN_DECLS
#ifdef	__cplusplus
#define __BEGIN_DECLS	extern "C" {
#define __END_DECLS		}
#else
#define __BEGIN_DECLS
#define __END_DECLS
#endif
#endif


STATIC_ASSERT(sizeof(u8) == 1);
STATIC_ASSERT(sizeof(u16) == 2);
STATIC_ASSERT(sizeof(u32) == 4);
STATIC_ASSERT(sizeof(u64) == 8);

typedef void*	Handle;

/*
* We define it here so that other platform code like hw_plat_mem.h can use the defines.
*/
enum HW_LOG_LEVEL
{
	HW_LOG_LEVEL_MIN = 0,
	HW_LOG_LEVEL_MINMINUSONE = HW_LOG_LEVEL_MIN - 1,

	/*
	* The macro is to control, not be set by output log.
	* You use it to set log level(output all log), you should never use the level to
	* output log, otherwise it will trigger common fatal while 1.
	*/
	HW_LOG_LEVEL_ALL,
	/*
	* Should be used only when debug mode.
	*/
	HW_LOG_LEVEL_DEBUG,
	/*
	* It is recommended that any log higher or equal to the trace log level should be written down
	* in the inner buffer, so that we can dump it out when occur err or fatal events.
	*/
	HW_LOG_LEVEL_TRACE,
	HW_LOG_LEVEL_INFO,
	HW_LOG_LEVEL_WARN,
	HW_LOG_LEVEL_ERR,
	/*
	* Log which cannot be masked. It should not be treated as error.
	*/
	HW_LOG_LEVEL_UNMASK,
	/*
	* The macro is to control, not be set by output log.
	* You use it to set log level(output no log except fatal), you should never use
	* the level to output log, otherwise it will trigger common fatal while 1.
	*/
	HW_LOG_LEVEL_OFF,
	/*
	* The log of the type will never be masked.
	* Use the type to mean you should poll print the log and do not expect to continue running any code.
	*/
	HW_LOG_LEVEL_FATAL,

	HW_LOG_LEVEL_MAXADDONE,
	HW_LOG_LEVEL_MAX = HW_LOG_LEVEL_MAXADDONE - 1,
};


#define HW_TIMEOUT_FOREVER		(-1U)
/*
* We you set HW_TIMEOUT_FOREVER as timeout value, the system will use the HW_TIMEOUT_US_DEFAULT
* as once timeout wait value. It will duplicate according to the timeout wait return value.
*/
#define HW_TIMEOUT_US_DEFAULT	(1000000U)

/*
* Get bits h:low of u64 u.
*/
#define U64BITH2L(u, h, low)		((((u64)(u))>>(low))&(((h)-(low)+1==64)?(~0):(~((u64)-1<<((h)-(low)+1)))))
#define U32BITH2L(u, h, low)		((((u32)(u))>>(low))&(((h)-(low)+1==32)?(~0):(~((u32)-1<<((h)-(low)+1)))))
#define U16BITH2L(u, h, low)		((((u16)(u))>>(low))&(((h)-(low)+1==16)?(~0):(~((u16)-1<<((h)-(low)+1)))))
#define U8BITH2L(u, h, low)			((((u8)(u))>>(low))&(((h)-(low)+1==8)?(~0):(~((u8)-1<<((h)-(low)+1)))))

/*
* 1 means is 2~N
* 0 means is not 2~N
*/
#define ISTWOPOWER(v)				(!((v) & ((v)-1)))

/*
* Common macro for performance enhance.
*/
#define HW_LIKELY(cond)				__glibc_likely((cond))
#define HW_UNLIKELY(cond)			__glibc_unlikely((cond))

#define HW_UNREFERENCE(param)		(param)

#endif
