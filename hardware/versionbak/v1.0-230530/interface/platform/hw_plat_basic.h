#ifndef HW_PLAT_BASIC_H
#define HW_PLAT_BASIC_H

#include "hw_plat_basedef.h"

__BEGIN_DECLS

typedef struct hw_atomic_u32_t
{
	void* parray_pvoid[4];
} hw_atomic_u32_t;

STATIC_ASSERT(sizeof(struct hw_atomic_u32_t) >= HW_PLAT_ERG_BYTECOUNT);

typedef struct hw_atomic_u64_t
{
	void* parray_pvoid[4];
} hw_atomic_u64_t;

STATIC_ASSERT(sizeof(struct hw_atomic_u64_t) >= HW_PLAT_ERG_BYTECOUNT);

typedef struct hw_atomic_pvoid_t
{
	void* parray_pvoid[4];
} hw_atomic_pvoid_t;

STATIC_ASSERT(sizeof(struct hw_atomic_pvoid_t) >= HW_PLAT_ERG_BYTECOUNT);

typedef struct hw_time_t
{
	u64		sec;
	u64		nsec;
} hw_time_t;

typedef struct hw_mutex_t
{
	void* parray_pvoid[8];
} hw_mutex_t;

STATIC_ASSERT(sizeof(struct hw_mutex_t) >= HW_PLAT_ERG_BYTECOUNT);

typedef struct hw_event_t
{
	void* parray_pvoid[16];
} hw_event_t;

STATIC_ASSERT(sizeof(struct hw_mutex_t) >= HW_PLAT_ERG_BYTECOUNT);

typedef struct hw_multievent_t
{
	void* parray_pvoid[16];
} hw_multievent_t;

STATIC_ASSERT(sizeof(struct hw_multievent_t) >= HW_PLAT_ERG_BYTECOUNT);

enum HW_MUTEX_TYPE
{
	HW_MUTEX_TYPE_MIN = 0,
	HW_MUTEX_TYPE_MINMINUSONE = HW_MUTEX_TYPE_MIN - 1,

	/*
	* Single process, can be recursively called.
	*/
	HW_MUTEX_TYPE_PROCESS_PRIVATE_RECURSIVE,

	HW_MUTEX_TYPE_MAXADDONE,
	HW_MUTEX_TYPE_MAX = HW_MUTEX_TYPE_MAXADDONE - 1,
};
#define HW_MUTEX_TYPE_DEFAULT			HW_MUTEX_TYPE_PROCESS_PRIVATE_RECURSIVE

// used by hw_plat_mutex_timedlock only
enum HW_MUTEX_RETSTATUS
{
	HW_MUTEX_RETSTATUS_MIN = 0,
	HW_MUTEX_RETSTATUS_MINMINUSONE = HW_MUTEX_RETSTATUS_MIN - 1,

	HW_MUTEX_RETSTATUS_LOCK,
	HW_MUTEX_RETSTATUS_TIMEOUT,

	HW_MUTEX_RETSTATUS_MAXADDONE,
	HW_MUTEX_RETSTATUS_MAX = HW_MUTEX_RETSTATUS_MAXADDONE - 1,
};

enum HW_EVENT_TYPE
{
	HW_EVENT_TYPE_MIN = 0,
	HW_EVENT_TYPE_MINMINUSONE = HW_EVENT_TYPE_MIN - 1,

	HW_EVENT_TYPE_AUTORESET_PROCESS_PRIVATE,
	HW_EVENT_TYPE_MANUALRESET_PROCESS_PRIVATE,

	HW_EVENT_TYPE_MAXADDONE,
	HW_EVENT_TYPE_MAX = HW_EVENT_TYPE_MAXADDONE - 1,
};

// used by hw_plat_event_timedwait only
enum HW_EVENT_RETSTATUS
{
	HW_EVENT_RETSTATUS_MIN = 0,
	HW_EVENT_RETSTATUS_MINMINUSONE = HW_EVENT_RETSTATUS_MIN - 1,

	HW_EVENT_RETSTATUS_SIGNAL,
	HW_EVENT_RETSTATUS_TIMEOUT,

	HW_EVENT_RETSTATUS_MAXADDONE,
	HW_EVENT_RETSTATUS_MAX = HW_EVENT_RETSTATUS_MAXADDONE - 1,
};

enum HW_MULTIEVENT_TYPE
{
	HW_MULTIEVENT_TYPE_MIN = 0,
	HW_MULTIEVENT_TYPE_MINMINUSONE = HW_MULTIEVENT_TYPE_MIN - 1,

	HW_MULTIEVENT_TYPE_AUTORESET,
	HW_MULTIEVENT_TYPE_MANUALRESET,

	HW_MULTIEVENT_TYPE_MAXADDONE,
	HW_MULTIEVENT_TYPE_MAX = HW_MULTIEVENT_TYPE_MAXADDONE - 1,
};

// used by hw_plat_multievent_timedwait only
enum HW_MULTIEVENT_RETSTATUS
{
	HW_MULTIEVENT_RETSTATUS_MIN = 0,
	HW_MULTIEVENT_RETSTATUS_MINMINUSONE = HW_MULTIEVENT_RETSTATUS_MIN - 1,

	HW_MULTIEVENT_RETSTATUS_SIGNAL,
	HW_MULTIEVENT_RETSTATUS_TIMEOUT,

	HW_MULTIEVENT_RETSTATUS_MAXADDONE,
	HW_MULTIEVENT_RETSTATUS_MAX = HW_MULTIEVENT_RETSTATUS_MAXADDONE - 1,
};

/*
* The least strict sync operation.
*/
s32 hw_plat_dmb();
/*
* It is stricter than hw_plat_dmb.
*/
s32 hw_plat_dsb();
/*
* It is the strictest sync operation. It will flush and reset the cpu pipeline.
*/
s32 hw_plat_isb();

/*
* The only difference between get is to add volatile.
* Used the relaxed memory order.
*/
s32 hw_plat_atomic_get_u32(struct hw_atomic_u32_t* io_patomic_u32, u32* o_pgetu32);
s32 hw_plat_atomic_get_u64(struct hw_atomic_u64_t* io_patomic_u64, u64* o_pgetu64);
/*
* *o_pgetpvoid is the pvoid value to get.
*/
s32 hw_plat_atomic_get_pvoid(struct hw_atomic_pvoid_t* io_patomic_pvoid, void** o_pgetpvoid);

/*
* The only difference between get is to add volatile.
*/
s32 hw_plat_atomic_set_u32(struct hw_atomic_u32_t* io_patomic_u32, u32 i_setu32);
s32 hw_plat_atomic_set_u64(struct hw_atomic_u64_t* io_patomic_u64, u64 i_setu64);
s32 hw_plat_atomic_set_pvoid(struct hw_atomic_pvoid_t* io_patomic_pvoid, void* i_setpvoid);

/*
* Always strong mode. When you need to use weak mode, use hw_plat_atomic_cas_exchange_weak_u32 instead.
* return 0 means exchange success.
* return <0 means exchange fail.
* You can set o_pu32now to NULL to mean you do not care about the current value after the operation.
* When o_pu32now is not NULL, *o_pu32now is i_u32new when return 0.
* When o_pu32now is not NULL, *o_pu32now is current value when return <0.
*/
s32 hw_plat_atomic_cas_exchange_u32(struct hw_atomic_u32_t* io_patomic_u32, u32 i_u32old, u32 i_u32new, u32* o_pu32now);
s32 hw_plat_atomic_cas_exchange_weak_u32(struct hw_atomic_u32_t* io_patomic_u32, u32 i_u32old, u32 i_u32new, u32* o_pu32now);
s32 hw_plat_atomic_cas_exchange_u64(struct hw_atomic_u64_t* io_patomic_u64, u64 i_u64old, u64 i_u64new, u64* o_pu64now);
s32 hw_plat_atomic_cas_exchange_weak_u64(struct hw_atomic_u64_t* io_patomic_u64, u64 i_u64old, u64 i_u64new, u64* o_pu64now);
s32 hw_plat_atomic_cas_exchange_pvoid(struct hw_atomic_pvoid_t* io_patomic_pvoid, void* i_pvoidold, void* i_pvoidnew, void** o_ppvoidnow);
s32 hw_plat_atomic_cas_exchange_weak_pvoid(struct hw_atomic_pvoid_t* io_patomic_pvoid, void* i_pvoidold, void* i_pvoidnew, void** o_ppvoidnow);

/*
* Should always return 0. (memory order, when device order you should not use cas operation)
* You can set o_pu32old or o_pu32old to NULL to mean you do not care about the old or new value.
*/
s32 hw_plat_atomic_cas_exchangeadd_u32(struct hw_atomic_u32_t* io_patomic_u32, u32 i_u32add, u32* o_pu32old, u32* o_pu32new);
s32 hw_plat_atomic_cas_exchangeadd_u64(struct hw_atomic_u64_t* io_patomic_u64, u64 i_u64add, u64* o_pu64old, u64* o_pu64new);

/*
* Should always return 0. (memory order, when device order you should not use cas operation)
* You can set o_pu32old or o_pu32old to NULL to mean you do not care about the old or new value.
*/
s32 hw_plat_atomic_cas_exchangeminus_u32(struct hw_atomic_u32_t* io_patomic_u32, u32 i_u32minus, u32* o_pu32old, u32* o_pu32new);
s32 hw_plat_atomic_cas_exchangeminus_u64(struct hw_atomic_u64_t* io_patomic_u64, u64 i_u64minus, u64* o_pu64old, u64* o_pu64new);

s32 hw_plat_mutex_init(struct hw_mutex_t* io_pmutex, HW_MUTEX_TYPE i_type);
s32 hw_plat_mutex_deinit(struct hw_mutex_t* io_pmutex);
/*
* Use mode HW_MUTEX_TYPE_DEFAULT as default.
*/
s32 hw_plat_mutex_init_bydefault(struct hw_mutex_t* io_pmutex);
s32 hw_plat_mutex_lock(struct hw_mutex_t* io_pmutex);
/*
* When you need to wait forever, please use hw_plat_mutex_lock instead,
* When i_ptime is NULL or *i_ptime's sec and nsec are 0, the function will try lock once.
* o_pretstatus should always be valid, *o_pretstatus is lock or timeout.
*/
s32 hw_plat_mutex_timedlock(struct hw_mutex_t* io_pmutex, struct hw_time_t* i_ptime, HW_MUTEX_RETSTATUS* o_pretstatus);
s32 hw_plat_mutex_unlock(struct hw_mutex_t* io_pmutex);

/*
* i_bset is 1 to mean the init status is 'signal'
* i_bset is 0 to mean the init status is 'not signal'
* When init status is 'signal', the event wait function will return directly.
*/
s32 hw_plat_event_init(struct hw_event_t* io_pevent, HW_EVENT_TYPE i_type, u32 i_bset);
// use HW_EVENT_TYPE_MANUALRESET_PROCESS_PRIVATE and i_bset is 0
s32 hw_plat_event_init_bydefault(struct hw_event_t* io_pevent);
s32 hw_plat_event_wait(struct hw_event_t* io_pevent);
/*
* When you need to wait forever, please use hw_plat_event_wait instead,
* When i_ptime is NULL or *i_ptime's sec and nsec are 0, the function will try wait once.
* o_pretstatus should always be valid, *o_pretstatus is signal or timeout.
*/
s32 hw_plat_event_timedwait(struct hw_event_t* io_pevent, struct hw_time_t* i_ptime, HW_EVENT_RETSTATUS* o_pretstatus);
s32 hw_plat_event_set(struct hw_event_t* io_pevent);
// you may not use the reset function when you set autoreset event type
s32 hw_plat_event_reset(struct hw_event_t* io_pevent);

/*
* You need to call hw_plat_multievent_deinit function when destruction, because it malloc 
* memory dynamically when init according to i_maxnumevents.
* You need to add one by one by hw_plat_multievent_add function.
* Only support process private.
*/
s32 hw_plat_multievent_init(struct hw_multievent_t* io_pmultievent, u32 i_maxnumevents);
s32 hw_plat_multievent_deinit(struct hw_multievent_t* io_pmultievent);
/*
* i_indexevent begin from 0, max value is i_maxnumevents-1.
* Properly use HW_EVENT_TYPE_MANUALRESET_PROCESS_PRIVATE and i_bset is 0.
* Not every add mulievent index will be wait, it just store in the inner array.
* Input the items to wait by hw_plat_multievent_wait i_parraybwait.
* The index item already is added, you should call hw_plat_multievent_delete before 
* call hw_plat_multievent_add.
*/
s32 hw_plat_multievent_add(struct hw_multievent_t* io_pmultievent, u32 i_indexevent, HW_MULTIEVENT_TYPE i_type, u32 i_bset);
s32 hw_plat_multievent_delete(struct hw_multievent_t* io_pmultievent, u32 i_indexevent);
/*
* Wait according to i_parraybwait input, the array number should be equal to i_maxnumevents.
* We assume the array number is i_maxnumevents, so we do not need to input the array number.
* o_pindexevent should always be valid.
* When return 0, *o_pindexevent output the signal event index.
* The function will ignore the event index when it is not added.
*/
s32 hw_plat_multievent_wait(struct hw_multievent_t* io_pmultievent, u32* i_parraybwait, u32* o_pindexevent);
/*
* Wait according to i_parraybwait input, the array number should be equal to i_maxnumevents.
* We assume the array number is i_maxnumevents, so we do not need to input the array number.
* o_pindexevent and o_pretstatus should always be valid.
* When return 0, *o_pretstatus output signal or timeout, when is signal, *o_pindexevent output 
* the signal event index.
* The function will ignore the event index when it is not added.
* When you need to wait forever, please use hw_plat_multievent_wait instead,
* When i_ptime is NULL or *i_ptime's sec and nsec are 0, the function will try wait once.
*/
s32 hw_plat_multievent_timedwait(struct hw_multievent_t* io_pmultievent, u32* i_parraybwait, u32* o_pindexevent, 
	struct hw_time_t* i_ptime, HW_MULTIEVENT_RETSTATUS* o_pretstatus);
s32 hw_plat_multievent_set(struct hw_multievent_t* io_pmultievent, u32 i_indexevent);
s32 hw_plat_multievent_reset(struct hw_multievent_t* io_pmultievent, u32 i_indexevent);

#define HW_PLAT_LOCALTIME_DESCBYTECOUNT_MAX		64

/*
* Second precise.
* Time operation. io_pbuff should be at least HW_PLAT_LOCALTIME_DESCBYTECOUNT_MAX byte count.
* When o_pbytecount is not NULL, *o_pbytecount is the byte count of the string after format.
* The byte count do not include the '\0' end.
* You can set o_pbytecount to NULL to mean you do not care about the byte count.
*/
s32 hw_plat_localtime_getdescstr(char* io_pbuff, u32* o_pbytecount);

#define HW_PLAT_TICKTIME_DESCBYTECOUNT_MAX		32

/*
* Milli-second precise.
* Get time span from system start.
* The byte count do not include the '\0' end.
* You can set o_pbytecount to NULL to mean you do not care about the byte count.
*/
s32 hw_plat_ticktime_getdescstr(char* io_pbuff, u32* o_pbytecount);

/*
* File define. The hal platform will not provide file hal function.
* We define the structure here to tag the file context which may be used within the hal platform 
* implement.
*/
typedef struct hw_plat_filehandle_t
{
	void* parray_pvoid[4];
} hw_plat_filehandle_t;

/*
* Register(reg) the signal(sig) handler.
* Use the default implement. Currently, in linux it will register SIGSEGV, SIGABRT and SIGBUS.
* The handler will output the callstack.
*/
s32 hw_plat_regsighandler_default();

/*
* Unregister(reg) the signal(sig) handler.
* Correspondent to function hw_plat_regsighandler_default.
* Set the registered signal function back to system default.
*/
s32 hw_plat_unregsighandler_default();

__END_DECLS

#endif
