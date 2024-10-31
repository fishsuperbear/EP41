#include <unistd.h>
#include <errno.h>
#include <poll.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/prctl.h>
#include "sys_statemgr/include/tegra_hv_sysctl.h"
#include "sys_statemgr/include/sys_define.h"
#include "sys_statemgr/include/logger.h"

namespace hozon {
namespace netaos {
namespace ssm {

TegraHvSysCtl::TegraHvSysCtl(){
    m_stopFlag = 0;
	m_fd = 0;
}

TegraHvSysCtl::~TegraHvSysCtl(){}

void TegraHvSysCtl::DeInit(){
    m_stopFlag = 1;
    if(m_thr_1.joinable()){ m_thr_1.join(); }
	if(m_thr_2.joinable()){ m_thr_2.join(); }
    if(m_thr_3.joinable()){ m_thr_3.join(); }
}

int32_t TegraHvSysCtl::Init(){
	m_fd = open(HV_PM_CTL_PATH, O_RDWR);
	if (m_fd < 0) {
		SSM_LOG_ERROR <<"hv_pm_ctl_init failed to open "<<HV_PM_CTL_PATH<<",ecode "<< -errno;
        return -1; 
	}
	return 0;
}

void TegraHvSysCtl::Run(){
	m_thr_1 = std::thread(&TegraHvSysCtl::HVPowerModeMonitor,this);
	m_thr_2 = std::thread(&TegraHvSysCtl::SysMsgHandle,this);
	m_thr_3 = std::thread(&TegraHvSysCtl::SSMHeartbeat,this);
}

void TegraHvSysCtl::HVPowerModeMonitor(void *arg){
	std::string thr_name = "thr_monitor";
    prctl(PR_SET_NAME, thr_name.c_str());
	TegraHvSysCtl *sys = static_cast<TegraHvSysCtl *>(arg);
	struct pollfd fds;
	struct hv_sysmgr_message msg;
	unsigned int read_mask = POLLIN | POLLPRI;
	unsigned int error_mask = POLLHUP;
	int ret = 0;
	fds.fd = sys->m_fd;
	fds.events = read_mask;
	while (!sys->m_stopFlag) {
		ret = poll(&fds, 1, 1000);
		if (ret < 0) {
			SSM_LOG_WARN <<"failed to poll "<< HV_PM_CTL_PATH <<",ret "<< ret;
		} else {
			if (fds.revents & error_mask) {
				SSM_LOG_WARN <<"error occurred on poll "<< HV_PM_CTL_PATH << ",0x" <<std::hex <<fds.revents;
			} else if (fds.revents & read_mask) {
				ret = sys->PMCtrlRecvMsg(&msg);
				if (ret < 0) {
					SSM_LOG_WARN <<"failed to receive msg,ret " << ret;
					continue;
				}
                sys->HVSysmsgPreProces(msg);
			}
		}
	}
}

void TegraHvSysCtl::HVSysmsgPreProces(struct hv_sysmgr_message & msg){
    struct hv_sysmgr_message hv_msg;
	hv_msg.msg_type = msg.msg_type;
	hv_msg.socket_id = msg.socket_id;
	memcpy(hv_msg.client_data, msg.client_data, SYSMGR_IVCMSG_SIZE_MAX);
    struct hv_sysmgr_command *cmd = (struct hv_sysmgr_command *)&(msg.client_data[0]);
	SSM_LOG_DEBUG << "pm ctrl cmd:"<<cmd->cmd_id;

	if (msg.msg_type == HV_SYSMGR_MSG_TYPE_GUEST_EVENT) {
        switch (cmd->cmd_id & 0xff) {
            case 0x11:
            case 0x12:
			    EnqueueSSMsg(hv_msg);
			    break;
			default:
			    SSM_LOG_WARN << "invalid msg type";
			    break;
		}
	}
    if (msg.msg_type == HV_SYSMGR_MSG_TYPE_VM_PM_CTL_CMD){
        EnqueueSSMsg(hv_msg);
	}
}

void TegraHvSysCtl::MCUMsgHandle(void *arg){
}

void TegraHvSysCtl::SysMsgHandle(void *arg){
	std::string thr_name = "thr_msghandle";
    prctl(PR_SET_NAME, thr_name.c_str());
    TegraHvSysCtl *sys = static_cast<TegraHvSysCtl *>(arg);
    while (!sys->m_stopFlag) {
		if(!sys->IsQueueEmpty()){
			struct hv_sysmgr_message msg;
            sys->DequeueSSMsg(&msg);
			struct hv_sysmgr_command *cmd = (struct hv_sysmgr_command *)&(msg.client_data[0]);
            int32_t ret = 0; 
		    switch (cmd->cmd_id) {
		    case HV_SYSMGR_CMD_NORMAL_SHUTDOWN:
			    SSM_LOG_INFO << "PM_CTL CMD, NORMAL_SHUTDOWN";
			    cmd->resp_id = HV_SYSMGR_RESP_ACCEPTED;
			    ret = sys->PMCtrlSendMsg(&msg);
				if(!ret){
			        ret = sys->PMCtrlReboot(true);
				}
			    break;
		    case HV_SYSMGR_CMD_NORMAL_REBOOT:
			    SSM_LOG_INFO << "PM_CTL CMD, NORMAL_REBOOT";
			    cmd->resp_id = HV_SYSMGR_RESP_ACCEPTED;
			    ret = sys->PMCtrlSendMsg(&msg);
				if(!ret){
			        ret = sys->PMCtrlReboot(false);
				}
			    break;
		    case HV_SYSMGR_CMD_NORMAL_SUSPEND:
			    SSM_LOG_INFO <<"PM_CTL CMD, NORMAL_SUSPEND";
			    cmd->resp_id = HV_SYSMGR_RESP_ACCEPTED;
			    ret = sys->PMCtrlSendMsg(&msg);
				if(!ret){
			        ret = sys->PMCtrlSuspend();
				}

			    break;
		    case HV_SYSMGR_CMD_NORMAL_RESUME:
			    SSM_LOG_INFO << "PM_C CMD, NORMAL_RESUME";
			    cmd->resp_id = HV_SYSMGR_RESP_ACCEPTED;
			    ret = sys->PMCtrlSendMsg(&msg);
				if(ret){
			        ret = sys->PMCtrlResume();
				}
			    break;

            case NOS_SYSMGR_CMD_SOC_REQUEST_RESET:
                SSM_LOG_INFO << "PM CMD, NORMAL_RESET";
			    break;
            case NOS_SYSMGR_CMD_SOC_REQUEST_RESTART:
                SSM_LOG_INFO << "PM CMD, NORMAL_RESTART";
			    break;	
		    default:
			    SSM_LOG_INFO << "unsupported PM_C CMD, 0x" << cmd->cmd_id;
			    cmd->resp_id = HV_SYSMGR_RESP_UNKNOWN_COMMAND;
			    ret = sys->PMCtrlSendMsg(&msg);
			    ret = -EINVAL;
			    break;
		    }
			if(ret){ SSM_LOG_ERROR << "< hv pm ctl fail >";}
	    }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
	}
}

void TegraHvSysCtl::SSMHeartbeat(void *arg){
	std::string thr_name = "thr_heartbeat";
    prctl(PR_SET_NAME, thr_name.c_str());
	TegraHvSysCtl *sys = static_cast<TegraHvSysCtl *>(arg);
	// struct hv_sysmgr_message msg;
    // msg.msg_type = HV_SYSMGR_MSG_TYPE_GUEST_EVENT;
	// msg.socket_id = sys->m_fd;
    // msg.client_data[0] = nos_sysmgr_cmd_id::NOS_SYSMGR_CMD_SOC_HEARTBEAT;
	while (!sys->m_stopFlag) {
        std::this_thread::sleep_for(std::chrono::seconds(1u));
    }
}


int32_t TegraHvSysCtl::PMCtrlRecvMsg(struct hv_sysmgr_message *msg){
	int32_t ret;
	ret = read(m_fd, msg, sizeof(*msg));
	if (ret == -1) {
		SSM_LOG_ERROR <<"failed to read data,ecode" <<-errno;
		ret = -errno;
	} else if (ret != sizeof(*msg)) {
		SSM_LOG_ERROR << "reading is not completed,read size "<< ret <<",expected size " << (int)sizeof(*msg);
		ret = -EIO;
	}
	return 0;
}

int32_t TegraHvSysCtl::PMCtrlSendMsg(struct hv_sysmgr_message *msg){
	int32_t ret;
	ret = write(m_fd, msg, sizeof(*msg));
	if (ret == -1) {
		SSM_LOG_ERROR << "failed to write data" << -errno;
		ret = -errno;
	} else if (ret != sizeof(*msg)) {
		SSM_LOG_ERROR << "writing is not completed,written size "<< ret <<",expected size" << (int)sizeof(*msg);
		ret = -EIO;
	}
	return 0;
}

int32_t TegraHvSysCtl::PMCtrlReboot(bool is_shutdown){
	int32_t ret = 0;
	if (is_shutdown)
		ret = system(HV_PM_CTL_SHUTDOWN);
	else
		ret = system(HV_PM_CTL_REBOOT);
	if(ret < 0) {
		SSM_LOG_CRITICAL << "failed to run cmd " << (is_shutdown ? HV_PM_CTL_SHUTDOWN : HV_PM_CTL_REBOOT);
	}
	return ret;
}

int32_t TegraHvSysCtl::PMCtrlSuspend(void){
	int32_t ret;
	ret = system(HV_PM_CTL_SUSPEND);
	if(ret < 0) {
		SSM_LOG_CRITICAL << "failed to run cmd " << HV_PM_CTL_SUSPEND;
	}
	return ret;
}

int32_t TegraHvSysCtl::PMCtrlResume(void){
    return 0;
}


void TegraHvSysCtl::EmptyQueueSSMsg(){
    std::lock_guard<std::mutex> lock(m_mutex_sstate);
    if(!m_que_sstate.empty()){
        std::queue<hv_sysmgr_message> empty;
        swap(empty,m_que_sstate);
    }
}

void TegraHvSysCtl::EnqueueSSMsg(struct hv_sysmgr_message &msg){
    std::lock_guard<std::mutex> lock(m_mutex_sstate);
    if(m_que_sstate.size() > 50){
        SSM_LOG_CRITICAL <<"queue is blocked";
    }else{
        m_que_sstate.push(msg);
        if(m_que_sstate.size() > 2){
            SSM_LOG_WARN <<"queue size:"<<m_que_sstate.size();
        }
    }
}

bool TegraHvSysCtl::IsQueueEmpty(){
    std::lock_guard<std::mutex> lock(m_mutex_sstate);
	return m_que_sstate.empty();
}

void TegraHvSysCtl::DequeueSSMsg(struct hv_sysmgr_message *msg){
    std::lock_guard<std::mutex> lock(m_mutex_sstate);
    msg = (hv_sysmgr_message *) &m_que_sstate.front();
    m_que_sstate.pop();
}

}}}