#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "neta_dbg_manager.h"
#include "zmq_ipc/manager/zmq_ipc_client.h"

using namespace std;
using namespace hozon::netaos::sm;
using namespace hozon::netaos::em;
using namespace hozon::netaos::log;
using namespace hozon::netaos::dbg;

int main(int argc, char*argv[])
{
	hozon::netaos::log::InitLogging("smdbg","sm dbg",hozon::netaos::log::LogLevel::kOff,
			hozon::netaos::log::HZ_LOG2FILE , "/opt/usr/log/soc_log", 10, 20);

    std::shared_ptr<NetaDbgManager> dbg(new NetaDbgManager());
	dbg->Init();

    switch (argc)
	{
	case 0:
    case 1:
	    dbg->Help();
		break;
	case 2:{
		if(strcmp("-h",argv[1]) == 0 || strcmp("-H",argv[1]) == 0){
            dbg->Help();
		}else if(strcmp("-v",argv[1]) == 0){
            printf("%s\n",DBG_VERSION);
		}else if (strcmp("reboot",argv[1]) == 0){
            dbg->Reboot();
		}else if (strcmp("reset",argv[1]) == 0){
            dbg->Reset();
        }else if (strcmp("list",argv[1]) == 0){
            dbg->GetModeListDetailInfo();
        }
	}
	    break;
	case 3:{
        if(strcmp("restart",argv[1]) == 0){
			std::string pname = argv[2];
            // size_t pos = pname.find("Process");
            // if(pos != pname.npos){
            dbg->RestartProcess(pname);
			// }
		}else if (strcmp("query",argv[1]) == 0){
            std::string param = argv[2];
			if(param == "modeList"){
                dbg->GetModeList();
			}else if (param == "processStatus"){
                dbg->GetProcessState();
			}
		}
	}
        break;
    case 4:{
        if (strcmp("request",argv[1]) == 0 && strcmp("mode",argv[2]) == 0){
            std::string mode = argv[3];
			if(mode.size() > 0){
			    if(mode == "Off"){
                    dbg->StopMode();
				}else{
                    dbg->SwitchMode(mode);
				}
			}
		}else  if (strcmp("set",argv[1]) == 0 && strcmp("startupMode",argv[2]) == 0){
            std::string mode = argv[3];
			if(mode.size() > 0){
                dbg->SetDefaultStartupMode(mode);
			}
		}
	}
        break;
	default:
	    dbg->Help();
		break;
	}

    dbg->DeInit();
    return 0;
}
