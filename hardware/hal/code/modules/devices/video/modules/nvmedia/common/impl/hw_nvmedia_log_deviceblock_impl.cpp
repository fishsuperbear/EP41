#include "hw_nvmedia_log_deviceblock_impl.h"

static struct hw_impl_nvmediadeviceblocklogenv _nvmediadeviceblocklogenv =
{
	.binit = 0,
};

static s32 hw_nvmedia_impl_log_deviceblock_init()
{
	if (HW_LIKELY(_nvmediadeviceblocklogenv.binit == 0))
	{
		s32 ret;
		ret = hw_plat_logcontext_fill_bydefault(&_nvmediadeviceblocklogenv.logcontext);
		if (ret < 0) {
			return -1;
		}
		strcpy(_nvmediadeviceblocklogenv.logcontext.logoper.innerimpl.logdir, "./hallog/nvmedia/deviceblock");
		u32 initvalue = 0;
		ret = hw_plat_logcontext_fill_bufmode_logbuf(&_nvmediadeviceblocklogenv.logcontext,
			_nvmediadeviceblocklogenv.logringbuffer, HW_NVMEDIA_DEVICEBLOCK_IMPL_LOGRINGBUFFER_BYTECOUNT, HW_PLAT_LOGCONTEXT_LOGBUFLEVEL_DEFAULT,
			&_nvmediadeviceblocklogenv.atomic_offset, &initvalue);
		if (ret < 0) {
			return -1;
		}
		ret = hw_plat_logcontext_init(&_nvmediadeviceblocklogenv.logcontext);
		if (ret < 0) {
			return -1;
		}
		_nvmediadeviceblocklogenv.binit = 1;
		return 0;
	}
	return -1;
}

struct hw_plat_logcontext_t* internal_get_plogcontext_nvmedia_deviceblock()
{
	if (HW_UNLIKELY(_nvmediadeviceblocklogenv.binit == 0))
	{
		hw_nvmedia_impl_log_deviceblock_init();
	}
	return &_nvmediadeviceblocklogenv.logcontext;
}
