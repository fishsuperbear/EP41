#include "hw_nvmedia_def_impl.h"

const char* hw_ret_s32_getdesc(hw_ret_s32 i_s32)
{
	if (HW_UNLIKELY(!HW_RET_S32_IS_CUSTOM(i_s32)))
	{
		/*
		* If is not custom, we cannot parse it.
		*/
		return "NA_not_custom";
	}
	switch (HW_RET_S32_GET_FACILITY(i_s32))
	{
	case HW_RET_S32_FACILITY_HW_HAL_NVMEDIA:
		switch (HW_RET_S32_GET_CODE(i_s32))
		{
		case HW_RET_S32_CODE_HW_HAL_NVMEDIA_COMMON_UNEXPECTED:
			return "HW_HAL_NVMEDIA_common_unexpected";
		case HW_RET_S32_CODE_HW_HAL_NVMEDIA_CHECK_VERSION_FAIL:
			return "HW_HAL_NVMEDIA_check_version_fail";
		case HW_RET_S32_CODE_HW_HAL_NVMEDIA_PIPELINE_CONFIG_WRONG:
			return "HW_HAL_NVMEDIA_pipeline_config_wrong";
		case HW_RET_S32_CODE_HW_HAL_NVMEDIA_BLOCKPIPELINE_NOT_ALWAYS_ENABLE_INNERHANDLE:
			return "HW_HAL_NVMEDIA_blockpipeline_not_always_enable_innerhandle";
		case HW_RET_S32_CODE_HW_HAL_NVMEDIA_SENSORPIPELINE_NOT_ALWAYS_ENABLE_INNERHANDLE:
			return "HW_HAL_NVMEDIA_sensorpipeline_not_always_enable_innerhandle";
		case HW_RET_S32_CODE_HW_HAL_NVMEDIA_DEVICE_ALREADY_IN_USE:
			return "HW_HAL_NVMEDIA_device_already_in_use";
		case HW_RET_S32_CODE_HW_HAL_NVMEDIA_DEVICE_CLOSE_EXCHANGE_FAIL:
			return "HW_HAL_NVMEDIA_device_close_exchange_fail";
		case HW_RET_S32_CODE_HW_HAL_NVMEDIA_PIPELINE_ALREADY_IN_USE:
			return "HW_HAL_NVMEDIA_pipeline_already_in_use";
		case HW_RET_S32_CODE_HW_HAL_NVMEDIA_PIPELINE_CLOSE_EXCHANGE_FAIL:
			return "HW_HAL_NVMEDIA_pipeline_close_exchange_fail";
		case HW_RET_S32_CODE_HW_HAL_NVMEDIA_CONNECT_DID_NOT_RECEIVE_CONNECTED_EVENT:
			return "HW_HAL_NVMEDIA_connect_did_not_receive_connected_event";
		case HW_RET_S32_CODE_HW_HAL_NVMEDIA_RECONCILE_EVENTHANDLER_NOT_RUNNING:
			return "HW_HAL_NVMEDIA_reconcile_eventhandler_not_running";
		case HW_RET_S32_CODE_HW_HAL_NVMEDIA_BCANCALLPIPELINESTART_UNEXPECTED_NOT_0:
			return "HW_HAL_NVMEDIA_bcancallpipelinestart_unexpected_not_0";
		case HW_RET_S32_CODE_HW_HAL_NVMEDIA_BCANCALLPIPELINESTART_UNEXPECTED_NOT_1:
			return "HW_HAL_NVMEDIA_bcancallpipelinestart_unexpected_not_1";
		case HW_RET_S32_CODE_HW_HAL_NVMEDIA_REGDATACBTYPE_UNEXPECTED:
			return "HW_HAL_NVMEDIA_regdatacbtype_unexpected";
		case HW_RET_S32_CODE_HW_HAL_NVMEDIA_REGDATACBTYPE_REGISTER_TWICE:
			return "HW_HAL_NVMEDIA_regdatacbtype_register_twice";
		case HW_RET_S32_CODE_HW_HAL_NVMEDIA_NO_CORRESPONDENT_OUTPUTTYPE:
			return "HW_HAL_NVMEDIA_no_correspondent_outputtype";
		case HW_RET_S32_CODE_HW_HAL_NVMEDIA_OUTPUTTYPE_REGISTER_TWICE:
			return "HW_HAL_NVMEDIA_outputtype_register_twice";
		case HW_RET_S32_CODE_HW_HAL_NVMEDIA_OUTPUTTYPE_NOT_ENABLE_BUT_REGISTER_DATACB:
			return "HW_HAL_NVMEDIA_outputtype_not_enable_but_register_datacb";
		case HW_RET_S32_CODE_HW_HAL_NVMEDIA_ONLY_SUPPORT_DIRECTCB_MODE:
			return "HW_HAL_NVMEDIA_only_support_directcb_mode";
		case HW_RET_S32_CODE_HW_HAL_NVMEDIA_ONLY_SUPPORT_SYNCCB_MODE:
			return "HW_HAL_NVMEDIA_only_support_synccb_mode";
		case HW_RET_S32_CODE_HW_HAL_NVMEDIA_COMMON_CONSUMER_SUBTYPE_NOT_EXPECTED:
			return "HW_HAL_NVMEDIA_common_consumer_subtype_not_expected";
		case HW_RET_S32_CODE_HW_HAL_NVMEDIA_COMMON_CONSUMER_UNKNOWN_SUBTYPE:
			return "HW_HAL_NVMEDIA_common_consumer_unknown_subtype";
		default:
			return "NA_HAL_NVMEDIA_unknown_code";
		}
		break;
	case HW_RET_S32_FACILITY_NVMEDIA_SIPLSTATUS:
		switch (HW_RET_S32_GET_CODE(i_s32))
		{
		case NVSIPL_STATUS_BAD_ARGUMENT:
			return "NVMEDIA_SIPLSTATUS_bad_argument";
		case NVSIPL_STATUS_NOT_SUPPORTED:
			return "NVMEDIA_SIPLSTATUS_not_supported";
		case NVSIPL_STATUS_OUT_OF_MEMORY:
			return "NVMEDIA_SIPLSTATUS_out_of_memory";
		case NVSIPL_STATUS_RESOURCE_ERROR:
			return "NVMEDIA_SIPLSTATUS_resource_error";
		case NVSIPL_STATUS_TIMED_OUT:
			return "NVMEDIA_SIPLSTATUS_timed_out";
		case NVSIPL_STATUS_INVALID_STATE:
			return "NVMEDIA_SIPLSTATUS_invalid_state";
		case NVSIPL_STATUS_EOF:
			return "NVMEDIA_SIPLSTATUS_eof";
		case NVSIPL_STATUS_NOT_INITIALIZED:
			return "NVMEDIA_SIPLSTATUS_not_initialized";
		case NVSIPL_STATUS_FAULT_STATE:
			return "NVMEDIA_SIPLSTATUS_fault_state";
		case NVSIPL_STATUS_ERROR:
			return "NVMEDIA_SIPLSTATUS_error";
		default:
			return "NA_NVMEDIA_SIPLSTATUS_unknown_code";
		}
		break;
	case HW_RET_S32_FACILITY_NVMEDIA_NVSCISTATUS:
		switch (HW_RET_S32_GET_CODE(i_s32))
		{
		case NvSciError_Unknown:
			return "NVMEDIA_NVSCISTATUS_unknown";
		case NvSciError_NotImplemented:
			return "NVMEDIA_NVSCISTATUS_notimplemented";
		case NvSciError_NotSupported:
			return "NVMEDIA_NVSCISTATUS_notsupported";
		case NvSciError_AccessDenied:
			return "NVMEDIA_NVSCISTATUS_accessdenied";
		case NvSciError_NotPermitted:
			return "NVMEDIA_NVSCISTATUS_notpermitted";
		case NvSciError_InvalidState:
			return "NVMEDIA_NVSCISTATUS_invalidstate";
		case NvSciError_InvalidOperation:
			return "NVMEDIA_NVSCISTATUS_invalidoperation";
		case NvSciError_NotInitialized:
			return "NVMEDIA_NVSCISTATUS_notinitialized";
		case NvSciError_AlreadyInUse:
			return "NVMEDIA_NVSCISTATUS_alreadyinuse";
		case NvSciError_AlreadyDone:
			return "NVMEDIA_NVSCISTATUS_alreadydone";
		case NvSciError_NotYetAvailable:
			return "NVMEDIA_NVSCISTATUS_notyetavailable";
		case NvSciError_NoLongerAvailable:
			return "NVMEDIA_NVSCISTATUS_nolongeravailable";
		case NvSciError_InsufficientMemory:
			return "NVMEDIA_NVSCISTATUS_insufficientmemory";
		case NvSciError_InsufficientResource:
			return "NVMEDIA_NVSCISTATUS_insufficientresource";
		case NvSciError_ResourceError:
			return "NVMEDIA_NVSCISTATUS_resourceerror";
		case NvSciError_BadParameter:
			return "NVMEDIA_NVSCISTATUS_badparameter";
		case NvSciError_BadAddress:
			return "NVMEDIA_NVSCISTATUS_badaddress";
		case NvSciError_TooBig:
			return "NVMEDIA_NVSCISTATUS_toobig";
		case NvSciError_Overflow:
			return "NVMEDIA_NVSCISTATUS_overflow";
		case NvSciError_InconsistentData:
			return "NVMEDIA_NVSCISTATUS_inconsistentdata";
		case NvSciError_InsufficientData:
			return "NVMEDIA_NVSCISTATUS_insufficientdata";
		case NvSciError_IndexOutOfRange:
			return "NVMEDIA_NVSCISTATUS_indexoutofrange";
		case NvSciError_ValueOutOfRange:
			return "NVMEDIA_NVSCISTATUS_valueoutofrange";
		case NvSciError_Timeout:
			return "NVMEDIA_NVSCISTATUS_timeout";
		case NvSciError_TryItAgain:
			return "NVMEDIA_NVSCISTATUS_tryitagain";
		case NvSciError_Busy:
			return "NVMEDIA_NVSCISTATUS_busy";
		case NvSciError_InterruptedCall:
			return "NVMEDIA_NVSCISTATUS_interruptedcall";
		case NvSciError_NoSuchDevice:
			return "NVMEDIA_NVSCISTATUS_nosuchdevice";
		case NvSciError_NoSpace:
			return "NVMEDIA_NVSCISTATUS_nospace";
		case NvSciError_NoSuchDevAddr:
			return "NVMEDIA_NVSCISTATUS_nosuchdevaddr";
		case NvSciError_IO:
			return "NVMEDIA_NVSCISTATUS_io";
		case NvSciError_InvalidIoctlNum:
			return "NVMEDIA_NVSCISTATUS_invalidioctlnum";
		case NvSciError_NoSuchEntry:
			return "NVMEDIA_NVSCISTATUS_nosuchentry";
		case NvSciError_BadFileDesc:
			return "NVMEDIA_NVSCISTATUS_badfiledesc";
		case NvSciError_CorruptedFileSys:
			return "NVMEDIA_NVSCISTATUS_corruptedfilesys";
		case NvSciError_FileExists:
			return "NVMEDIA_NVSCISTATUS_fileexists";
		case NvSciError_IsDirectory:
			return "NVMEDIA_NVSCISTATUS_isdirectory";
		case NvSciError_ReadOnlyFileSys:
			return "NVMEDIA_NVSCISTATUS_readonlyfilesys";
		case NvSciError_TextFileBusy:
			return "NVMEDIA_NVSCISTATUS_textfilebusy";
		case NvSciError_FileNameTooLong:
			return "NVMEDIA_NVSCISTATUS_filenametoolong";
		case NvSciError_FileTooBig:
			return "NVMEDIA_NVSCISTATUS_filetoobig";
		case NvSciError_TooManySymbolLinks:
			return "NVMEDIA_NVSCISTATUS_toomanysymbollinks";
		case NvSciError_TooManyOpenFiles:
			return "NVMEDIA_NVSCISTATUS_toomanyopenfiles";
		case NvSciError_FileTableOverflow:
			return "NVMEDIA_NVSCISTATUS_filetableoverflow";
		case NvSciError_EndOfFile:
			return "NVMEDIA_NVSCISTATUS_endoffile";
		case NvSciError_ConnectionReset:
			return "NVMEDIA_NVSCISTATUS_connectionreset";
		case NvSciError_AlreadyInProgress:
			return "NVMEDIA_NVSCISTATUS_alreadyinprogress";
		case NvSciError_NoData:
			return "NVMEDIA_NVSCISTATUS_nodata";
		case NvSciError_NoDesiredMessage:
			return "NVMEDIA_NVSCISTATUS_nodesiredmessage";
		case NvSciError_MessageSize:
			return "NVMEDIA_NVSCISTATUS_messagesize";
		case NvSciError_NoRemote:
			return "NVMEDIA_NVSCISTATUS_noremote";
		case NvSciError_NoSuchProcess:
			return "NVMEDIA_NVSCISTATUS_nosuchprocess";
		case NvSciError_MutexNotRecoverable:
			return "NVMEDIA_NVSCISTATUS_mutexnotrecoverable";
		case NvSciError_LockOwnerDead:
			return "NVMEDIA_NVSCISTATUS_lockownerdead";
		case NvSciError_ResourceDeadlock:
			return "NVMEDIA_NVSCISTATUS_resourcedeadlock";
		case NvSciError_ReconciliationFailed:
			return "NVMEDIA_NVSCISTATUS_reconciliationfailed";
		case NvSciError_AttrListValidationFailed:
			return "NVMEDIA_NVSCISTATUS_attrlistvalidationfailed";
		case NvSciError_CommonEnd:
			return "NVMEDIA_NVSCISTATUS_commonend";
		case NvSciError_NvSciBufUnknown:
			return "NVMEDIA_NVSCISTATUS_nvscibufunknown";
		case NvSciError_NvSciBufEnd:
			return "NVMEDIA_NVSCISTATUS_nvscibufend";
		case NvSciError_NvSciSyncUnknown:
			return "NVMEDIA_NVSCISTATUS_nvscisyncunknown";
		case NvSciError_UnsupportedConfig:
			return "NVMEDIA_NVSCISTATUS_unsupportedconfig";
		case NvSciError_ClearedFence:
			return "NVMEDIA_NVSCISTATUS_clearedfence";
		case NvSciError_NvSciSyncEnd:
			return "NVMEDIA_NVSCISTATUS_nvscisyncend";
		case NvSciError_NvSciStreamUnknown:
			return "NVMEDIA_NVSCISTATUS_nvscistreamunknown";
		case NvSciError_StreamInternalError:
			return "NVMEDIA_NVSCISTATUS_streaminternalerror";
		case NvSciError_StreamBadBlock:
			return "NVMEDIA_NVSCISTATUS_streambadblock";
		case NvSciError_StreamBadPacket:
			return "NVMEDIA_NVSCISTATUS_streambadpacket";
		case NvSciError_StreamBadCookie:
			return "NVMEDIA_NVSCISTATUS_streambadcookie";
		case NvSciError_StreamNotConnected:
			return "NVMEDIA_NVSCISTATUS_streamnotconnected";
		case NvSciError_StreamNotSetupPhase:
			return "NVMEDIA_NVSCISTATUS_streamnotsetupphase";
		case NvSciError_StreamNotSafetyPhase:
			return "NVMEDIA_NVSCISTATUS_streamnotsafetyphase";
		case NvSciError_NoStreamPacket:
			return "NVMEDIA_NVSCISTATUS_nostreampacket";
		case NvSciError_StreamPacketInaccessible:
			return "NVMEDIA_NVSCISTATUS_streampacketinaccessiable";
		case NvSciError_StreamPacketDeleted:
			return "NVMEDIA_NVSCISTATUS_streampacketdeleted";
		case NvSciError_StreamInfoNotProvided:
			return "NVMEDIA_NVSCISTATUS_streaminfonotprovided";
		case NvSciError_StreamLockFailed:
			return "NVMEDIA_NVSCISTATUS_streamlockfailed";
		case NvSciError_StreamBadSrcIndex:
			return "NVMEDIA_NVSCISTATUS_streambadsrcindex";
		case NvSciError_StreamBadDstIndex:
			return "NVMEDIA_NVSCISTATUS_streambaddstindex";
		case NvSciError_NvSciStreamEnd:
			return "NVMEDIA_NVSCISTATUS_nvscistreamend";
		case NvSciError_NvSciIpcUnknown:
			return "NVMEDIA_NVSCISTATUS_nvsciipcunknown";
		case NvSciError_NvSciIpcEnd:
			return "NVMEDIA_NVSCISTATUS_nvsciipcend";
		case NvSciError_NvSciEventUnknown:
			return "NVMEDIA_NVSCISTATUS_nvscieventunknown";
		case NvSciError_NvSciEventEnd:
			return "NVMEDIA_NVSCISTATUS_nvscieventend";
		default:
			return "NA_NVMEDIA_NVSCISTATUS_unknown_code";
		}
	default:
		return "NA_unknown_facility";
	}
}

void HWNvmediaDeviceOpenPara::Init(HW_NVMEDIA_APPTYPE i_apptype, const char* i_platformname, const char* i_maskstr,
	u32 i_busemailbox, const char* i_nitofolderpath)
{
	apptype = i_apptype;
	platformname = i_platformname;
	maskstr = i_maskstr;
	char maskstrtemp[128];
	strcpy(maskstrtemp, i_maskstr);
	char* saveptr;
	char* token = __strtok_r((char*)maskstrtemp, " ", &saveptr);
	while (token != NULL) {
		vmasks.push_back(std::stoi(token, nullptr, 16));
		token = __strtok_r(NULL, " ", &saveptr);
	}
	busemailbox = i_busemailbox;
	nitofolderpath = i_nitofolderpath;
}
