#ifndef HW_IMPL_PLAT_BASIC_H
#define HW_IMPL_PLAT_BASIC_H

#include "hw_impl_plat_baseinc.h"

struct hw_impl_atomic_u32
{
	std::atomic<u32>				data;
};

STATIC_ASSERT(sizeof(hw_impl_atomic_u32) <= sizeof(hw_atomic_u32_t));

struct hw_impl_atomic_u64
{
	std::atomic<u64>				data;
};

STATIC_ASSERT(sizeof(hw_impl_atomic_u64) <= sizeof(hw_atomic_u64_t));

struct hw_impl_atomic_pvoid
{
	std::atomic<void*>				data;
};

STATIC_ASSERT(sizeof(hw_impl_atomic_pvoid) <= sizeof(hw_atomic_pvoid_t));

struct hw_impl_mutex
{
	pthread_mutex_t					mutex;
};

STATIC_ASSERT(sizeof(hw_impl_mutex) <= sizeof(hw_mutex_t));

struct hw_impl_event
{
	pthread_mutex_t					mutex;
	pthread_cond_t					cond;
	// 0 or 1, 0 mean not signal, 1 mean signal
	u32								bset;
	// 0 or 1, 0 mean need manual reset, 1 mean reset immediately once wait return
	u32								bautoreset;
};

STATIC_ASSERT(sizeof(hw_impl_event) <= sizeof(hw_event_t));

struct hw_impl_multieventsetinfo
{
	/*
	* 1 mean has used hw_plat_multievent_add to add event to the index
	* 0 mean has not add event to the index or use hw_plat_multievent_delete to delete
	* the index
	*/
	u32								badded;
	u32								bset;	// see note in hw_impl_event
	u32								bautoreset;	// see note in hw_impl_event
};

struct hw_impl_multievent
{
	pthread_mutex_t					mutex;
	pthread_cond_t					cond;
	u32								arraynumsetinfo;
	hw_impl_multieventsetinfo*		parraysetinfo;
};

STATIC_ASSERT(sizeof(hw_impl_multievent) <= sizeof(hw_multievent_t));

struct hw_impl_plat_filecontrol
{
	FILE*							fp;
	u32								bcasadd;
	/*
	* Valid when bcasadd is 1.
	* Invalid when bcasadd is 0.
	*/
	struct hw_atomic_u32_t			atomic_u32;
};

struct hw_impl_plat_file
{
	hw_impl_plat_filecontrol*		pcontrol;
};

STATIC_ASSERT(sizeof(hw_impl_plat_file) <= sizeof(hw_plat_filehandle_t));

#endif
