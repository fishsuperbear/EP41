#include "hw_impl_plat_basic.h"
#include "hw_impl_plat_logdump.h"

s32 hw_plat_dmb()
{
	asm volatile("dmb sy" : : : "memory");
	return 0;
}

s32 hw_plat_dsb()
{
	asm volatile("dsb sy" : : : "memory");
	return 0;
}

s32 hw_plat_isb()
{
	asm volatile("isb sy" : : : "memory");
	return 0;
}

s32 hw_plat_atomic_get_u32(struct hw_atomic_u32_t* io_patomic_u32, u32* o_pgetu32)
{
	hw_impl_atomic_u32* pimplu32 = (hw_impl_atomic_u32*)io_patomic_u32;
	*o_pgetu32 = pimplu32->data.load(std::memory_order_relaxed);
	return 0;
}

s32 hw_plat_atomic_get_u64(struct hw_atomic_u64_t* io_patomic_u64, u64* o_pgetu64)
{
	hw_impl_atomic_u64* pimplu64 = (hw_impl_atomic_u64*)io_patomic_u64;
	*o_pgetu64 = pimplu64->data.load(std::memory_order_relaxed);
	return 0;
}

s32 hw_plat_atomic_get_pvoid(struct hw_atomic_pvoid_t* io_patomic_pvoid, void** o_pgetpvoid)
{
	hw_impl_atomic_pvoid* pimplpvoid = (hw_impl_atomic_pvoid*)io_patomic_pvoid;
	*o_pgetpvoid = pimplpvoid->data.load(std::memory_order_relaxed);
	return 0;
}

s32 hw_plat_atomic_set_u32(struct hw_atomic_u32_t* io_patomic_u32, u32 i_setu32)
{
	hw_impl_atomic_u32* pimplu32 = (hw_impl_atomic_u32*)io_patomic_u32;
	pimplu32->data.store(i_setu32, std::memory_order_relaxed);
	return 0;
}

s32 hw_plat_atomic_set_u64(struct hw_atomic_u64_t* io_patomic_u64, u64 i_setu64)
{
	hw_impl_atomic_u64* pimplu64 = (hw_impl_atomic_u64*)io_patomic_u64;
	pimplu64->data.store(i_setu64, std::memory_order_relaxed);
	return 0;
}

s32 hw_plat_atomic_set_pvoid(struct hw_atomic_pvoid_t* io_patomic_pvoid, void* i_setpvoid)
{
	hw_impl_atomic_pvoid* pimplpvoid = (hw_impl_atomic_pvoid*)io_patomic_pvoid;
	pimplpvoid->data.store(i_setpvoid, std::memory_order_relaxed);
	return 0;
}

s32 hw_plat_atomic_cas_exchange_u32(struct hw_atomic_u32_t* io_patomic_u32, u32 i_u32old, u32 i_u32new, u32* o_pu32now)
{
	hw_impl_atomic_u32* pimplu32 = (hw_impl_atomic_u32*)io_patomic_u32;
	u32 u32now = i_u32old;
	if (pimplu32->data.compare_exchange_strong(u32now, i_u32new, std::memory_order_relaxed))
	{
		if (o_pu32now) {
			*o_pu32now = i_u32new;
		}
		return 0;
	}
	if (o_pu32now) {
		*o_pu32now = u32now;
	}
	return -1;
}

s32 hw_plat_atomic_cas_exchange_weak_u32(struct hw_atomic_u32_t* io_patomic_u32, u32 i_u32old, u32 i_u32new, u32* o_pu32now)
{
	hw_impl_atomic_u32* pimplu32 = (hw_impl_atomic_u32*)io_patomic_u32;
	u32 u32now = i_u32old;
	if (pimplu32->data.compare_exchange_weak(u32now, i_u32new, std::memory_order_relaxed))
	{
		if (o_pu32now) {
			*o_pu32now = i_u32new;
		}
		return 0;
	}
	if (o_pu32now) {
		*o_pu32now = u32now;
	}
	return -1;
}

s32 hw_plat_atomic_cas_exchange_u64(struct hw_atomic_u64_t* io_patomic_u64, u64 i_u64old, u64 i_u64new, u64* o_pu64now)
{
	hw_impl_atomic_u64* pimplu64 = (hw_impl_atomic_u64*)io_patomic_u64;
	u64 u64now = i_u64old;
	if (pimplu64->data.compare_exchange_strong(u64now, i_u64new, std::memory_order_relaxed))
	{
		if (o_pu64now) {
			*o_pu64now = i_u64new;
		}
		return 0;
	}
	if (o_pu64now) {
		*o_pu64now = u64now;
	}
	return -1;
}

s32 hw_plat_atomic_cas_exchange_weak_u64(struct hw_atomic_u64_t* io_patomic_u64, u64 i_u64old, u64 i_u64new, u64* o_pu64now)
{
	hw_impl_atomic_u64* pimplu64 = (hw_impl_atomic_u64*)io_patomic_u64;
	u64 u64now = i_u64old;
	if (pimplu64->data.compare_exchange_weak(u64now, i_u64new, std::memory_order_relaxed))
	{
		if (o_pu64now) {
			*o_pu64now = i_u64new;
		}
		return 0;
	}
	if (o_pu64now) {
		*o_pu64now = u64now;
	}
	return -1;
}

s32 hw_plat_atomic_cas_exchange_pvoid(struct hw_atomic_pvoid_t* io_patomic_pvoid, void* i_pvoidold, void* i_pvoidnew, void** o_ppvoidnow)
{
	hw_impl_atomic_pvoid* pimplpvoid = (hw_impl_atomic_pvoid*)io_patomic_pvoid;
	void* pvoidnow = i_pvoidold;
	if (pimplpvoid->data.compare_exchange_strong(pvoidnow, i_pvoidnew, std::memory_order_relaxed))
	{
		if (o_ppvoidnow) {
			*o_ppvoidnow = i_pvoidnew;
		}
		return 0;
	}
	if (o_ppvoidnow) {
		*o_ppvoidnow = pvoidnow;
	}
	return -1;
}

s32 hw_plat_atomic_cas_exchange_weak_pvoid(struct hw_atomic_pvoid_t* io_patomic_pvoid, void* i_pvoidold, void* i_pvoidnew, void** o_ppvoidnow)
{
	hw_impl_atomic_pvoid* pimplpvoid = (hw_impl_atomic_pvoid*)io_patomic_pvoid;
	void* pvoidnow = i_pvoidold;
	if (pimplpvoid->data.compare_exchange_weak(pvoidnow, i_pvoidnew, std::memory_order_relaxed))
	{
		if (o_ppvoidnow) {
			*o_ppvoidnow = i_pvoidnew;
		}
		return 0;
	}
	if (o_ppvoidnow) {
		*o_ppvoidnow = pvoidnow;
	}
	return -1;
}

s32 hw_plat_atomic_cas_exchangeadd_u32(struct hw_atomic_u32_t* io_patomic_u32, u32 i_u32add, u32* o_pu32old, u32* o_pu32new)
{
	hw_impl_atomic_u32* pimplu32 = (hw_impl_atomic_u32*)io_patomic_u32;
	u32 u32old = pimplu32->data.fetch_add(i_u32add, std::memory_order_relaxed);
	if (o_pu32old) {
		*o_pu32old = u32old;
	}
	if (o_pu32new) {
		*o_pu32new = u32old + i_u32add;
	}
	return 0;
}

s32 hw_plat_atomic_cas_exchangeadd_u64(struct hw_atomic_u64_t* io_patomic_u64, u64 i_u64add, u64* o_pu64old, u64* o_pu64new)
{
	hw_impl_atomic_u64* pimplu64 = (hw_impl_atomic_u64*)io_patomic_u64;
	u64 u64old = pimplu64->data.fetch_add(i_u64add, std::memory_order_relaxed);
	if (o_pu64old) {
		*o_pu64old = u64old;
	}
	if (o_pu64new) {
		*o_pu64new = u64old + i_u64add;
	}
	return 0;
}

s32 hw_plat_atomic_cas_exchangeminus_u32(struct hw_atomic_u32_t* io_patomic_u32, u32 i_u32minus, u32* o_pu32old, u32* o_pu32new)
{
	hw_impl_atomic_u32* pimplu32 = (hw_impl_atomic_u32*)io_patomic_u32;
	u32 u32old = pimplu32->data.fetch_sub(i_u32minus, std::memory_order_relaxed);
	if (o_pu32old) {
		*o_pu32old = u32old;
	}
	if (o_pu32new) {
		*o_pu32new = u32old - i_u32minus;
	}
	return 0;
}

s32 hw_plat_atomic_cas_exchangeminus_u64(struct hw_atomic_u64_t* io_patomic_u64, u64 i_u64minus, u64* o_pu64old, u64* o_pu64new)
{
	hw_impl_atomic_u64* pimplu64 = (hw_impl_atomic_u64*)io_patomic_u64;
	u64 u64old = pimplu64->data.fetch_sub(i_u64minus, std::memory_order_relaxed);
	if (o_pu64old) {
		*o_pu64old = u64old;
	}
	if (o_pu64new) {
		*o_pu64new = u64old - i_u64minus;
	}
	return 0;
}

s32 hw_plat_mutex_init(struct hw_mutex_t* io_pmutex, HW_MUTEX_TYPE i_type)
{
	if (i_type != HW_MUTEX_TYPE_PROCESS_PRIVATE_RECURSIVE)
	{
		hw_impl_plat_loginner_output("Do not support mode except HW_MUTEX_TYPE_PROCESS_PRIVATE_RECURSIVE!\r\n");
		hw_impl_plat_loginner_outputoff();
		return -1;
	}
	hw_impl_mutex* pmutex = (hw_impl_mutex*)io_pmutex;
	pthread_mutexattr_t mattr;
	pthread_mutexattr_init(&mattr);
	pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_PRIVATE);
	pthread_mutexattr_settype(&mattr, PTHREAD_MUTEX_RECURSIVE_NP);
	pthread_mutex_init(&pmutex->mutex, &mattr);
	return 0;
}

s32 hw_plat_mutex_deinit(struct hw_mutex_t* io_pmutex)
{
	hw_impl_mutex* pmutex = (hw_impl_mutex*)io_pmutex;
	pthread_mutex_destroy(&pmutex->mutex);
	return 0;
}

s32 hw_plat_mutex_init_bydefault(struct hw_mutex_t* io_pmutex)
{
	hw_impl_mutex* pmutex = (hw_impl_mutex*)io_pmutex;
	pthread_mutexattr_t mattr;
	pthread_mutexattr_init(&mattr);
	pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_PRIVATE);
	pthread_mutexattr_settype(&mattr, PTHREAD_MUTEX_RECURSIVE_NP);
	pthread_mutex_init(&pmutex->mutex, &mattr);
	return 0;
}

s32 hw_plat_mutex_lock(struct hw_mutex_t* io_pmutex)
{
	hw_impl_mutex* pmutex = (hw_impl_mutex*)io_pmutex;
	s32 ret = pthread_mutex_lock(&pmutex->mutex);
	if (HW_UNLIKELY(ret == EINVAL))
	{
		hw_impl_plat_loginner_output("hw_plat_mutex_lock pthread_mutex_lock EINVAL!\r\n");
			hw_impl_plat_logdump_backtrace_without_checkoutputstage();
			hw_impl_plat_loginner_outputoff();
		return -1;
	}
	if (HW_UNLIKELY(ret == EAGAIN))
	{
		hw_impl_plat_loginner_output("hw_plat_mutex_lock pthread_mutex_lock EAGAIN!\r\n");
			hw_impl_plat_logdump_backtrace_without_checkoutputstage();
			hw_impl_plat_loginner_outputoff();
		return -1;
	}
	return 0;
}

s32 hw_plat_mutex_timedlock(struct hw_mutex_t* io_pmutex, struct hw_time_t* i_ptime, HW_MUTEX_RETSTATUS* o_pretstatus)
{
	hw_impl_mutex* pmutex = (hw_impl_mutex*)io_pmutex;
	s32 ret = 0;
	struct timespec time_out;
	if (i_ptime != NULL)
	{
		if (i_ptime->sec == 0 && i_ptime->nsec == 0)
		{
			ret = pthread_mutex_trylock(&pmutex->mutex);
		}
		else
		{

			clock_gettime(CLOCK_REALTIME, &time_out);
			time_out.tv_sec += i_ptime->sec;
			time_out.tv_nsec += i_ptime->nsec;
			if (time_out.tv_nsec >= 1000000000)
			{
				time_out.tv_sec += 1;
				time_out.tv_nsec -= 1000000000;
				if (HW_UNLIKELY(time_out.tv_nsec >= 1000000000))
				{
					time_out.tv_sec += (time_out.tv_nsec / 1000000000);
					time_out.tv_nsec = time_out.tv_nsec % 1000000000;
				}
			}
			pthread_mutex_timedlock(&pmutex->mutex, &time_out);
		}
	}
	else
	{
		ret = pthread_mutex_trylock(&pmutex->mutex);
	}
	if (HW_LIKELY(ret == 0))
	{
		*o_pretstatus = HW_MUTEX_RETSTATUS_LOCK;
		return 0;
	}
	if (HW_LIKELY(ret == EBUSY))
	{
		*o_pretstatus = HW_MUTEX_RETSTATUS_TIMEOUT;
		return 0;
	}
	if (HW_UNLIKELY(ret == EINVAL))
	{
		hw_impl_plat_loginner_output("hw_plat_mutex_timedlock pthread_mutex_lock EINVAL!\r\n");
		hw_impl_plat_logdump_backtrace_without_checkoutputstage();
		hw_impl_plat_loginner_outputoff();
		return -1;
	}
	if (HW_UNLIKELY(ret == EAGAIN))
	{
		hw_impl_plat_loginner_output("hw_plat_mutex_timedlock pthread_mutex_lock EAGAIN!\r\n");
		hw_impl_plat_logdump_backtrace_without_checkoutputstage();
		hw_impl_plat_loginner_outputoff();
		return -1;
	}
	return 0;
}

s32 hw_plat_mutex_unlock(struct hw_mutex_t* io_pmutex)
{
	hw_impl_mutex* pmutex = (hw_impl_mutex*)io_pmutex;
	pthread_mutex_unlock(&pmutex->mutex);
	return 0;
}

s32 hw_plat_event_init(struct hw_event_t* io_pevent, HW_EVENT_TYPE i_type, u32 i_bset)
{
	hw_impl_event* pevent = (hw_impl_event*)io_pevent;
	pthread_mutexattr_t mattr;
	pthread_condattr_t condattr;
	pthread_mutexattr_init(&mattr);
	pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_PRIVATE);
	pthread_mutexattr_settype(&mattr, PTHREAD_MUTEX_RECURSIVE_NP);
	pthread_mutex_init(&pevent->mutex, &mattr);
	pthread_mutex_lock(&pevent->mutex);
	pthread_condattr_init(&condattr);
	pthread_condattr_setclock(&condattr, CLOCK_MONOTONIC);
	pthread_cond_init(&pevent->cond, &condattr);
	pevent->bset = i_bset;
	pevent->bautoreset = ((i_type == HW_EVENT_TYPE_AUTORESET_PROCESS_PRIVATE) ? 1 : 0);
	pthread_mutex_unlock(&pevent->mutex);
	return 0;
}

s32 hw_plat_event_wait(struct hw_event_t* io_pevent)
{
	hw_impl_event* pevent = (hw_impl_event*)io_pevent;
	pthread_mutex_lock(&pevent->mutex);
	while (1)
	{
		if (pevent->bset == 1)
		{
			/*
			* Signal status, need to reset to no signal if autoreset mode.
			*/
			if (pevent->bautoreset)
			{
				pevent->bset = 0;
			}
			pthread_mutex_unlock(&pevent->mutex);
			return 0;
		}
		pthread_cond_wait(&pevent->cond, &pevent->mutex);
	}
}

s32 hw_plat_event_timedwait(struct hw_event_t* io_pevent, struct hw_time_t* i_ptime, HW_EVENT_RETSTATUS* o_pretstatus)
{
	hw_impl_event* pevent = (hw_impl_event*)io_pevent;
	struct timespec time_out;
	pthread_mutex_lock(&pevent->mutex);
	if (i_ptime != NULL)
	{
		if (i_ptime->sec != 0 || i_ptime->nsec != 0)
		{
			if (pevent->bset == 1)
			{
				*o_pretstatus = HW_EVENT_RETSTATUS_SIGNAL;
				if (pevent->bautoreset)
				{
					pevent->bset = 0;
				}
				pthread_mutex_unlock(&pevent->mutex);
				return 0;
			}
			clock_gettime(CLOCK_MONOTONIC, &time_out);
			time_out.tv_sec += i_ptime->sec;
			time_out.tv_nsec += i_ptime->nsec;
			if (time_out.tv_nsec >= 1000000000)
			{
				time_out.tv_sec += 1;
				time_out.tv_nsec -= 1000000000;
				if (HW_UNLIKELY(time_out.tv_nsec >= 1000000000))
				{
					time_out.tv_sec += (time_out.tv_nsec / 1000000000);
					time_out.tv_nsec = time_out.tv_nsec % 1000000000;
				}
			}
			pthread_cond_timedwait(&pevent->cond, &pevent->mutex, &time_out);
		}
	}
	if (pevent->bset == 1)
	{
		*o_pretstatus = HW_EVENT_RETSTATUS_SIGNAL;
		if (pevent->bautoreset)
		{
			pevent->bset = 0;
		}
	}
	else
	{
		*o_pretstatus = HW_EVENT_RETSTATUS_TIMEOUT;
	}
	pthread_mutex_unlock(&pevent->mutex);
	return 0;
}

s32 hw_plat_event_set(struct hw_event_t* io_pevent)
{
	hw_impl_event* pevent = (hw_impl_event*)io_pevent;
	pthread_mutex_lock(&pevent->mutex);
	pevent->bset = 1;
	pthread_cond_signal(&pevent->cond);
	pthread_mutex_unlock(&pevent->mutex);
	return 0;
}

s32 hw_plat_event_reset(struct hw_event_t* io_pevent)
{
	hw_impl_event* pevent = (hw_impl_event*)io_pevent;
	pthread_mutex_lock(&pevent->mutex);
	pevent->bset = 0;
	pthread_mutex_unlock(&pevent->mutex);
	return 0;
}

s32 hw_plat_multievent_init(struct hw_multievent_t* io_pmultievent, u32 i_maxnumevents)
{
	hw_impl_multievent* pmultievent = (hw_impl_multievent*)io_pmultievent;
	pthread_mutexattr_t mattr;
	pthread_condattr_t condattr;
	u32 eventi;
	pmultievent->parraysetinfo = (hw_impl_multieventsetinfo*)malloc(sizeof(hw_impl_multieventsetinfo) * i_maxnumevents);
	for (eventi = 0; eventi < i_maxnumevents; eventi++)
	{
		pmultievent->parraysetinfo[eventi].badded = 0;
	}
	pthread_mutexattr_init(&mattr);
	pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_PRIVATE);
	pthread_mutexattr_settype(&mattr, PTHREAD_MUTEX_RECURSIVE_NP);
	pthread_mutex_init(&pmultievent->mutex, &mattr);
	pthread_mutex_lock(&pmultievent->mutex);
	pthread_condattr_init(&condattr);
	pthread_condattr_setclock(&condattr, CLOCK_MONOTONIC);
	pthread_cond_init(&pmultievent->cond, &condattr);
	pthread_mutex_unlock(&pmultievent->mutex);
	return 0;
}

s32 hw_plat_multievent_deinit(struct hw_multievent_t* io_pmultievent)
{
	hw_impl_multievent* pmultievent = (hw_impl_multievent*)io_pmultievent;
	pthread_mutex_lock(&pmultievent->mutex);
	pthread_cond_destroy(&pmultievent->cond);
	pthread_mutex_unlock(&pmultievent->mutex);
	pthread_mutex_destroy(&pmultievent->mutex);
	free(pmultievent->parraysetinfo);
	return 0;
}

s32 hw_plat_multievent_add(struct hw_multievent_t* io_pmultievent, u32 i_indexevent, HW_MULTIEVENT_TYPE i_type, u32 i_bset)
{
	hw_impl_multievent* pmultievent = (hw_impl_multievent*)io_pmultievent;
	hw_impl_multieventsetinfo* psetinfo;
	pthread_mutex_lock(&pmultievent->mutex);
	if (HW_UNLIKELY(i_indexevent >= pmultievent->arraynumsetinfo))
	{
		hw_impl_plat_loginner_output("hw_plat_multievent_add i_indexevent[%u] >= pmultievent->arraynumsetinfo[%u]!\r\n",
			i_indexevent, pmultievent->arraynumsetinfo);
		hw_impl_plat_logdump_backtrace_without_checkoutputstage();
		hw_impl_plat_loginner_outputoff();
		pthread_mutex_unlock(&pmultievent->mutex);
		return -1;
	}
	if (HW_UNLIKELY(pmultievent->parraysetinfo[i_indexevent].badded))
	{
		hw_impl_plat_loginner_output("hw_plat_multievent_add parraysetinfo i_indexevent[%u] already added!\r\n",
			i_indexevent);
		hw_impl_plat_logdump_backtrace_without_checkoutputstage();
		hw_impl_plat_loginner_outputoff();
		pthread_mutex_unlock(&pmultievent->mutex);
		return -1;
	}
	psetinfo = &pmultievent->parraysetinfo[i_indexevent];
	psetinfo->bautoreset = ((i_type == HW_MULTIEVENT_TYPE_AUTORESET) ? 1 : 0);
	psetinfo->bset = i_bset;
	psetinfo->badded = 1;
	pthread_mutex_unlock(&pmultievent->mutex);
	return 0;
}

s32 hw_plat_multievent_delete(struct hw_multievent_t* io_pmultievent, u32 i_indexevent)
{
	hw_impl_multievent* pmultievent = (hw_impl_multievent*)io_pmultievent;
	hw_impl_multieventsetinfo* psetinfo;
	pthread_mutex_lock(&pmultievent->mutex);
	if (HW_UNLIKELY(i_indexevent >= pmultievent->arraynumsetinfo))
	{
		hw_impl_plat_loginner_output("hw_plat_multievent_delete i_indexevent[%u] >= pmultievent->arraynumsetinfo[%u]!\r\n",
			i_indexevent, pmultievent->arraynumsetinfo);
		hw_impl_plat_logdump_backtrace_without_checkoutputstage();
		hw_impl_plat_loginner_outputoff();
		pthread_mutex_unlock(&pmultievent->mutex);
		return -1;
	}
	if (HW_UNLIKELY(pmultievent->parraysetinfo[i_indexevent].badded == 0))
	{
		hw_impl_plat_loginner_output("hw_plat_multievent_delete parraysetinfo i_indexevent[%u] NOT added!\r\n",
			i_indexevent);
		hw_impl_plat_logdump_backtrace_without_checkoutputstage();
		hw_impl_plat_loginner_outputoff();
		pthread_mutex_unlock(&pmultievent->mutex);
		return -1;
	}
	psetinfo = &pmultievent->parraysetinfo[i_indexevent];
	psetinfo->badded = 0;
	pthread_mutex_unlock(&pmultievent->mutex);
	return 0;
}

/*
* Already in lock before call the function.
* return -1 mean no signal
* return 0 mean has signal, *o_pindexevent is the signal event index.
*/
static s32 hw_impl_plat_multievent_checksignal(hw_impl_multievent* io_pmultievent, u32* i_parraybwait, u32* o_pindexevent)
{
	hw_impl_multievent* pmultievent = io_pmultievent;
	u32 eventi;
	hw_impl_multieventsetinfo* psetinfo;
	for (eventi = 0; eventi < pmultievent->arraynumsetinfo; eventi++)
	{
		psetinfo = &pmultievent->parraysetinfo[eventi];
		if (i_parraybwait[eventi] && psetinfo->badded)
		{
			if (psetinfo->bset)
			{
				*o_pindexevent = eventi;
				return 0;
			}
		}
	}
	return -1;
}

s32 hw_plat_multievent_wait(struct hw_multievent_t* io_pmultievent, u32* i_parraybwait, u32* o_pindexevent)
{
	hw_impl_multievent* pmultievent = (hw_impl_multievent*)io_pmultievent;
	hw_impl_multieventsetinfo* psetinfo;
	u32 eventi;
	s32 ret;
	for (eventi = 0; eventi < pmultievent->arraynumsetinfo; eventi++)
	{
		if (HW_UNLIKELY(i_parraybwait[eventi] != 0 && i_parraybwait[eventi] != 1))
		{
			hw_impl_plat_loginner_output("hw_plat_multievent_wait i_parraybwait[%u]=[%u] is not 0 or 1!\r\n",
				eventi, i_parraybwait[eventi]);
			hw_impl_plat_logdump_backtrace_without_checkoutputstage();
			hw_impl_plat_loginner_outputoff();
			return -1;
		}
	}
	pthread_mutex_lock(&pmultievent->mutex);
	while (1)
	{
		ret = hw_impl_plat_multievent_checksignal(pmultievent, i_parraybwait, o_pindexevent);
		if (ret == 0)
		{
			psetinfo = &pmultievent->parraysetinfo[*o_pindexevent];
			if (psetinfo->bautoreset)
			{
				psetinfo->bset = 0;
			}
			pthread_mutex_unlock(&pmultievent->mutex);
			return 0;
		}
		pthread_cond_wait(&pmultievent->cond, &pmultievent->mutex);
	}
}

s32 hw_plat_multievent_timedwait(struct hw_multievent_t* io_pmultievent, u32* i_parraybwait, u32* o_pindexevent,
	struct hw_time_t* i_ptime, HW_MULTIEVENT_RETSTATUS* o_pretstatus)
{
	hw_impl_multievent* pmultievent = (hw_impl_multievent*)io_pmultievent;
	hw_impl_multieventsetinfo* psetinfo;
	struct timespec time_out;
	s32 ret;
	pthread_mutex_lock(&pmultievent->mutex);
	if (i_ptime != NULL)
	{
		if (i_ptime->sec != 0 || i_ptime->nsec != 0)
		{
			ret = hw_impl_plat_multievent_checksignal(pmultievent, i_parraybwait, o_pindexevent);
			if (ret == 0)
			{
				*o_pretstatus = HW_MULTIEVENT_RETSTATUS_SIGNAL;
				psetinfo = &pmultievent->parraysetinfo[*o_pindexevent];
				if (psetinfo->bautoreset)
				{
					psetinfo->bset = 0;
				}
				pthread_mutex_unlock(&pmultievent->mutex);
				return 0;
			}
			clock_gettime(CLOCK_MONOTONIC, &time_out);
			time_out.tv_sec += i_ptime->sec;
			time_out.tv_nsec += i_ptime->nsec;
			if (time_out.tv_nsec >= 1000000000)
			{
				time_out.tv_sec += 1;
				time_out.tv_nsec -= 1000000000;
				if (HW_UNLIKELY(time_out.tv_nsec >= 1000000000))
				{
					time_out.tv_sec += (time_out.tv_nsec / 1000000000);
					time_out.tv_nsec = time_out.tv_nsec % 1000000000;
				}
			}
			pthread_cond_timedwait(&pmultievent->cond, &pmultievent->mutex, &time_out);
		}
	}
	ret = hw_impl_plat_multievent_checksignal(pmultievent, i_parraybwait, o_pindexevent);
	if (ret == 0)
	{
		*o_pretstatus = HW_MULTIEVENT_RETSTATUS_SIGNAL;
		psetinfo = &pmultievent->parraysetinfo[*o_pindexevent];
		if (psetinfo->bautoreset)
		{
			psetinfo->bset = 0;
		}
		pthread_mutex_unlock(&pmultievent->mutex);
		return 0;
	}
	*o_pretstatus = HW_MULTIEVENT_RETSTATUS_TIMEOUT;
	pthread_mutex_unlock(&pmultievent->mutex);
	return 0;
}

s32 hw_plat_multievent_set(struct hw_multievent_t* io_pmultievent, u32 i_indexevent)
{
	hw_impl_multievent* pmultievent = (hw_impl_multievent*)io_pmultievent;
	hw_impl_multieventsetinfo* psetinfo;
	pthread_mutex_lock(&pmultievent->mutex);
	if (HW_UNLIKELY(i_indexevent >= pmultievent->arraynumsetinfo))
	{
		hw_impl_plat_loginner_output("hw_plat_multievent_set i_indexevent[%u] >= pmultievent->arraynumsetinfo[%u]!\r\n",
			i_indexevent, pmultievent->arraynumsetinfo);
		hw_impl_plat_logdump_backtrace_without_checkoutputstage();
		hw_impl_plat_loginner_outputoff();
		pthread_mutex_unlock(&pmultievent->mutex);
		return -1;
	}
	if (HW_UNLIKELY(pmultievent->parraysetinfo[i_indexevent].badded == 0))
	{
		hw_impl_plat_loginner_output("hw_plat_multievent_set parraysetinfo i_indexevent[%u] NOT added!\r\n",
			i_indexevent);
		hw_impl_plat_logdump_backtrace_without_checkoutputstage();
		hw_impl_plat_loginner_outputoff();
		pthread_mutex_unlock(&pmultievent->mutex);
		return -1;
	}
	psetinfo = &pmultievent->parraysetinfo[i_indexevent];
	psetinfo->bset = 1;
	pthread_cond_signal(&pmultievent->cond);
	pthread_mutex_unlock(&pmultievent->mutex);
	return 0;
}

s32 hw_plat_multievent_reset(struct hw_multievent_t* io_pmultievent, u32 i_indexevent)
{
	hw_impl_multievent* pmultievent = (hw_impl_multievent*)io_pmultievent;
	hw_impl_multieventsetinfo* psetinfo;
	pthread_mutex_lock(&pmultievent->mutex);
	if (HW_UNLIKELY(i_indexevent >= pmultievent->arraynumsetinfo))
	{
		hw_impl_plat_loginner_output("hw_plat_multievent_reset i_indexevent[%u] >= pmultievent->arraynumsetinfo[%u]!\r\n",
			i_indexevent, pmultievent->arraynumsetinfo);
		hw_impl_plat_logdump_backtrace_without_checkoutputstage();
		hw_impl_plat_loginner_outputoff();
		pthread_mutex_unlock(&pmultievent->mutex);
		return -1;
	}
	if (HW_UNLIKELY(pmultievent->parraysetinfo[i_indexevent].badded == 0))
	{
		hw_impl_plat_loginner_output("hw_plat_multievent_reset parraysetinfo i_indexevent[%u] NOT added!\r\n",
			i_indexevent);
		hw_impl_plat_logdump_backtrace_without_checkoutputstage();
		hw_impl_plat_loginner_outputoff();
		pthread_mutex_unlock(&pmultievent->mutex);
		return -1;
	}
	psetinfo = &pmultievent->parraysetinfo[i_indexevent];
	psetinfo->bset = 0;
	pthread_mutex_unlock(&pmultievent->mutex);
	return 0;
}

s32 hw_plat_localtime_getdescstr(char* io_pbuff, u32* o_pbytecount)
{
	time_t timet;
	struct tm tmt;
	time(&timet);
	localtime_r(&timet, &tmt);
	u32 bytecount = sprintf(io_pbuff, "%d_%d_%d-%d_%d_%d",
		(1900 + tmt.tm_year), (1 + tmt.tm_mon), tmt.tm_mday,
		tmt.tm_hour, tmt.tm_min, tmt.tm_sec);
	if (o_pbytecount) {
		*o_pbytecount = bytecount;
	}
	return 0;
}

s32 hw_plat_ticktime_getdescstr(char* io_pbuff, u32* o_pbytecount)
{
	struct timespec timesp;
	clock_gettime(CLOCK_MONOTONIC, &timesp);
	u32 bytecount = sprintf(io_pbuff, "%6ld:%9ld", timesp.tv_sec, timesp.tv_nsec);
	if (o_pbytecount) {
		*o_pbytecount = bytecount;
	}
	return 0;
}

static void _hw_impl_plat_basic_sig_handler(s32 i_signum)
{
	hw_impl_plat_loginner_output_without_checkoutputstage("enter _hw_impl_plat_basic_sig_handler[%d]\r\n", i_signum);
	hw_impl_plat_logdump_backtrace();
	hw_impl_plat_loginner_output_without_checkoutputstage("finish logdump backtrace in _hw_impl_plat_basic_sig_handler[%d]\r\n", i_signum);
	hw_impl_plat_loginner_output_without_checkoutputstage("enable default sig handler[SIG_DFL] in _hw_impl_plat_basic_sig_handler[%d]\r\n", i_signum);
	hw_plat_logoutput_setoffstatus();
	signal(i_signum, SIG_DFL);
}

s32 hw_plat_regsighandler_default()
{
	signal(SIGSEGV, _hw_impl_plat_basic_sig_handler);
	signal(SIGABRT, _hw_impl_plat_basic_sig_handler);
	signal(SIGBUS, _hw_impl_plat_basic_sig_handler);
	return 0;
}

s32 hw_plat_unregsighandler_default()
{
	signal(SIGSEGV, SIG_DFL);
	signal(SIGABRT, SIG_DFL);
	signal(SIGBUS, SIG_DFL);
	return 0;
}

s32 hw_plat_get_tsc_ns(u64* io_tsc_ns)
{
    uint64_t tsc;
    __asm__ __volatile__ ("mrs %[tsc], cntvct_el0" : [tsc] "=r" (tsc));
    *io_tsc_ns = tsc * 32;
    return 0;
}
