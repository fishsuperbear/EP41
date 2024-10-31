#include "sys_statemgr/include/logger.h"
#include "sys_statemgr/include/sys_manager.h"
#include "sys_statemgr/include/function_statistics.h"

namespace hozon {
namespace netaos {
namespace ssm {

SysManager::SysManager() {}

SysManager::~SysManager() {}

void SysManager::DeInit() {
	SSM_LOG_INFO << __func__;
    sysstatemgrsvr->Stop();
	disptch->DeInit();
    msscli->DeInit();
    spmsvr->DeInit();
	ara::core::Deinitialize();
	statmgr->DeInit();
}

int32_t SysManager::Init(){
	SSM_LOG_INFO << __func__;
	statmgr = std::make_shared<StateManager>();
	statmgr->Init();

	disptch = std::make_shared<Dispatcher>();
	disptch->Init(statmgr);

    std::string instanceId= "1";
    ara::core::Initialize();

    spmsvr = std::make_shared<PowerManagerServiceSvr>(instanceId);
	spmsvr->Init(statmgr);

    msscli = std::make_shared<McuStateServiceCli>(instanceId);
	msscli->Init(statmgr);

    sysstatemgrsvr = std::make_shared<SysStateMgrServer>(this);

	return 0;
}

void SysManager::Run(){
	statmgr->Run();
	disptch->Run();
	spmsvr->Run();
    sysstatemgrsvr->Start();
}

std::shared_ptr<McuStateServiceCli> SysManager::McuStateService() {
    return msscli;
}

}}}
