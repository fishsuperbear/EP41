#include "hw_impl_plat_logdef.h"

static char _innerlog_path[HW_PLAT_LOGOUTPUT_INNER_PATH_BYTECOUNT_MAX] = HW_PLAT_LOGOUTPUT_INNER_PATH_DEFAULT;

/*
* 0 means has not output any inner log. Need to output the head string first.
* 1 means has output the head string. Can continue output string.
* <0 means cannot output any log again. Currently only -1 value.
*/
static s32 _outputstage = 0;

s32 hw_plat_innerlog_path_change(char* i_path)
{
	strcpy(_innerlog_path, i_path);
	return 0;
}

s32 hw_plat_logoutput_getonoffstatus(u32* o_ponstatus)
{
	if (HW_LIKELY(o_ponstatus != NULL))
	{
		if (_outputstage < 0)
		{
			*o_ponstatus = 0;
		}
		else
		{
			*o_ponstatus = 1;
		}
	}
	return 0;
}

s32 hw_plat_logoutput_setoffstatus()
{
	return hw_impl_plat_loginner_outputoff();
}

s32 hw_impl_plat_loginner_output(const char* i_pformat, ...)
{
	if (HW_LIKELY(_outputstage < 0)) {
		return -1;
	}
	va_list valist;
	va_start(valist, i_pformat);
	s32 ret = hw_impl_plat_loginner_output_valist_without_checkoutputstage(i_pformat, valist);
	va_end(valist);
	return ret;
}

s32 hw_impl_plat_loginner_output_without_checkoutputstage(const char* i_pformat, ...)
{
	va_list valist;
	va_start(valist, i_pformat);
	s32 ret = hw_impl_plat_loginner_output_valist_without_checkoutputstage(i_pformat, valist);
	va_end(valist);
	return ret;
}

s32 hw_impl_plat_loginner_output_valist_without_checkoutputstage(const char* i_pformat, va_list i_valist)
{
	FILE* fp = fopen(_innerlog_path, "a+");
	if (HW_UNLIKELY(fp == NULL)) {
		return -1;
	}
	char logbuf[HW_IMPL_PLAT_LOGOUTPUT_INNER_LOGBUF_BYTECOUNT_MAX];
	STATIC_ASSERT(HW_PLAT_LOCALTIME_DESCBYTECOUNT_MAX <= HW_IMPL_PLAT_LOGOUTPUT_INNER_LOGBUF_BYTECOUNT_MAX);
	u32 bytecount;
	if (HW_UNLIKELY(_outputstage == 0))
	{
		bytecount = sprintf(logbuf, "\r\n\r\n\r\nLog inner output...\r\n\r\nTime: ");
		fwrite(logbuf, bytecount, 1, fp);
		/*
		* Use mode 'a+' instead of mode 'a' so that we can read the file if needed.
		*/
		hw_plat_localtime_getdescstr(logbuf, &bytecount);
		fwrite(logbuf, bytecount, 1, fp);
		bytecount = sprintf(logbuf, "\r\n\r\n");
		fwrite(logbuf, bytecount, 1, fp);
		_outputstage = 1;
	}
	s32 vsnret = vsnprintf(logbuf, HW_IMPL_PLAT_LOGOUTPUT_INNER_LOGBUF_BYTECOUNT_MAX, i_pformat, i_valist);
	if (HW_UNLIKELY(vsnret < 0))
	{
		bytecount = sprintf(logbuf, "vsnprintf ret=%d,errno=%d\r\n", vsnret, errno);
		fwrite(logbuf, bytecount, 1, fp);
	}
	else
	{
		if (HW_LIKELY(vsnret < HW_IMPL_PLAT_LOGOUTPUT_INNER_LOGBUF_BYTECOUNT_MAX))
		{
			bytecount = vsnret;
		}
		else
		{
			bytecount = HW_IMPL_PLAT_LOGOUTPUT_INNER_LOGBUF_BYTECOUNT_MAX;
		}
		fwrite(logbuf, bytecount, 1, fp);
	}
	fflush(fp);
	fclose(fp);
	return 0;
}

s32 hw_impl_plat_loginner_outputoff()
{
	if (HW_LIKELY(_outputstage < 0)) {
		return -1;
	}
	/*
	* Change output stage to outputoff before other operation.
	* We should ensure the outputoff function be called only once.
	*/
	_outputstage = -1;
	/*
	* First output the stack.
	*/

	/*
	* Tag outputoff at the end.
	*/
	hw_impl_plat_loginner_output_without_checkoutputstage("\r\n\r\noutputoff\r\n\r\n");
	return 0;
}

s32 hw_impl_plat_system(char* i_pcmd, s32* o_pexitstatus)
{
	s32 ret;
	ret = system(i_pcmd);
	if (WIFEXITED(ret))
	{
		if (0 == WEXITSTATUS(ret))
		{
			/*
			* Do not output any log when run success.
			*/
			return 0;
		}
	}
	if (o_pexitstatus != NULL)
	{
		*o_pexitstatus = WEXITSTATUS(ret);
	}
	return -1;
}

s32 hw_impl_plat_file_open(struct hw_plat_filehandle_t* io_pfilehandle, char* i_filepath)
{
	hw_impl_plat_file* pimplfile = (hw_impl_plat_file*)io_pfilehandle;
	pimplfile->pcontrol = (hw_impl_plat_filecontrol*)malloc(sizeof(hw_impl_plat_filecontrol));
	pimplfile->pcontrol->fp = fopen(i_filepath, "w+");
	/*
	* Set not cas add as default.
	*/
	pimplfile->pcontrol->bcasadd = 0;
	return 0;
}

s32 hw_impl_plat_file_close(struct hw_plat_filehandle_t* io_pfilehandle)
{
	hw_impl_plat_file* pimplfile = (hw_impl_plat_file*)io_pfilehandle;
	fflush(pimplfile->pcontrol->fp);
	fclose(pimplfile->pcontrol->fp);
	free(pimplfile->pcontrol);
	return 0;
}

s32 hw_impl_plat_file_getfilep(struct hw_plat_filehandle_t* i_pfilehandle, FILE** o_ppfile)
{
	hw_impl_plat_file* pimplfile = (hw_impl_plat_file*)i_pfilehandle;
	*o_ppfile = pimplfile->pcontrol->fp;
	return 0;
}

s32 hw_impl_plat_file_setcasaddmode(struct hw_plat_filehandle_t* io_pfilehandle)
{
	hw_impl_plat_file* pimplfile = (hw_impl_plat_file*)io_pfilehandle;
	hw_plat_atomic_set_u32(&pimplfile->pcontrol->atomic_u32, 0);
	pimplfile->pcontrol->bcasadd = 1;
	return 1;
}
