#pragma once

#include <stdint.h>
#include <map>
#include <string>

namespace hozon {
namespace netaos {
namespace update {

struct common_req_t
{
	uint32_t platform;
};

struct update_status_resp_t
{
	std::string update_status;
	uint32_t error_code;
	std::string error_msg;
};
	
struct precheck_resp_t
{
	bool space;
	bool speed;
	bool gear;
	uint32_t error_code;
	std::string error_msg;
};
	
struct progress_resp_t
{
	uint32_t progress;
	uint32_t error_code;
	std::string error_msg;
};

struct start_update_req_t
{
	bool start_with_precheck;
	bool skip_version;
	// 0 : 升级大包所有件
	// 1 : 仅 升级 Soc + Mcu
	// 2 : 升级 LIDAR
	// 3 : 升级 SRR_FL
	// 4 : 升级 SRR_FR
	// 5 : 升级 SRR_RL
	// 6 : 升级 SRR_RR
	uint32_t ecu_mode;
	std::string package_path;
};
struct start_update_resp_t
{
	uint32_t error_code;
	std::string error_msg;
};

struct get_version_resp_t
{
	std::string major_version;
	std::string soc_version;
	std::string mcu_version;
	std::string dsv_version;
	std::string swt_version;
	std::map<std::string, std::string> sensor_version;
	uint32_t error_code;
	std::string error_msg;
};

struct start_finish_resp_t
{
	uint32_t error_code;
	std::string error_msg;
};

struct cur_pratition_resp_t
{
	std::string cur_partition;
	uint32_t error_code;
	std::string error_msg;
};

struct switch_slot_resp_t
{
	uint32_t error_code;
	std::string error_msg;
};

}  // namespace update
}  // namespace netaos
}  // namespace hozon

