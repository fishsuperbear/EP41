/**

* Copyright @ 2020 - 2027 iAuto Software(Shanghai) Co., Ltd.

* All Rights Reserved.

*

* Redistribution and use in source and binary forms, with or without

* modification, are NOT permitted except as agreed by

* iAuto Software(Shanghai) Co., Ltd.

*

* Unless required by applicable law or agreed to in writing, software

* distributed under the License is distributed on an "AS IS" BASIS,

* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

*/
#include "ne_someip_daemon.h"
#include "ne_someip_define.h"
#include "ne_someip_internal_define.h"
#include "ne_someip_log.h"
#include "ne_someip_ipc_daemon_behaviour.h"
#include "ne_someip_looper.h"
#include "ne_someip_thread.h"
#include "ne_someip_sd.h"
#include "ne_someip_sd_tool.h"
#include "ne_someip_daemon_internal.h"
#include "ne_someip_endpoint_unix.h"
#include "ne_someip_common_reuse_manager.h"

typedef struct ne_someip_client_id_info
{
	bool is_used;
	ne_someip_endpoint_unix_addr_t unix_addr;
} ne_someip_client_id_info_t;

struct ne_someip_daemon
{
	ne_someip_endpoint_unix_t* unix_endpoint;
	ne_someip_looper_t* io_looper;
	ne_someip_thread_t* io_thread;
	ne_someip_map_t* client_id_map;//<char*, ne_someip_client_id_info_t*>
};

static ne_someip_daemon_t* g_daemon = NULL;

#define NE_SOMEIP_CLIENT_ID_MIN 100
#define NE_SOMEIP_CLIENT_ID_MAX 10000

ne_someip_daemon_t* ne_someip_daemon_init()
{
	if (NULL == g_daemon) {
		g_daemon = (ne_someip_daemon_t*)malloc(sizeof(ne_someip_daemon_t));
		if (NULL == g_daemon) {
			ne_someip_log_error("malloc error");
			return NULL;
		}
	}
	memset(g_daemon, 0, sizeof(ne_someip_daemon_t));

	//start thread
	ne_someip_looper_config_t looper_config;
    g_daemon->io_looper = ne_someip_looper_new(&looper_config);
    if (NULL == g_daemon->io_looper) {
        ne_someip_log_error("g_daemon->io_looper new failed");
		if (NULL != g_daemon) {
            free(g_daemon);
			g_daemon = NULL;
		}
        return NULL;
    }

	//init sd
	bool b_ret = ne_someip_sd_init(g_daemon->io_looper);
	if (!b_ret) {
        ne_someip_log_error("sd init failed");
		ne_someip_looper_unref(g_daemon->io_looper);
		if (NULL != g_daemon) {
            free(g_daemon);
			g_daemon = NULL;
		}
		return NULL;
	}

    g_daemon->io_thread = ne_someip_thread_new_looper("io_thread", NULL, NULL, g_daemon->io_looper);
    if (NULL == g_daemon->io_thread) {
        ne_someip_log_error("g_daemon->io_thread new failed");
		ne_someip_sd_deinit();
        ne_someip_looper_unref(g_daemon->io_looper);
		if (NULL != g_daemon) {
            free(g_daemon);
			g_daemon = NULL;
		}
        return NULL;
    }

    int32_t c_ret = ne_someip_thread_start(g_daemon->io_thread);
    if (-1 == c_ret) {
		ne_someip_sd_deinit();
        ne_someip_looper_unref(g_daemon->io_looper);
        ne_someip_thread_unref(g_daemon->io_thread);
		if (NULL != g_daemon) {
            free(g_daemon);
			g_daemon = NULL;
		}
        return NULL;
    }

    //create unix endpoint
	g_daemon->unix_endpoint = ne_someip_ipc_daemon_behaviour_init(g_daemon->io_looper);
    if (NULL == g_daemon->unix_endpoint) {
		ne_someip_log_error("unix endpoint create failed");
        ne_someip_daemon_deinit();
		return NULL;
	}
	g_daemon->client_id_map = ne_someip_map_new(ne_someip_sd_uint16_hash_fun,
		ne_someip_sd_uint16_cmp, ne_someip_sd_free, ne_someip_sd_free);
    if (NULL == g_daemon->client_id_map) {
		ne_someip_log_error("map create failed");
        ne_someip_daemon_deinit();
		return NULL;
	}
	uint16_t tmp_id = NE_SOMEIP_CLIENT_ID_MIN;
	while (tmp_id <= NE_SOMEIP_CLIENT_ID_MAX) {
		ne_someip_client_id_t* tmp_client_id_key = (ne_someip_client_id_t*)malloc(sizeof(ne_someip_client_id_t));
		if (NULL == tmp_client_id_key) {
			ne_someip_log_error("malloc error");
			ne_someip_daemon_deinit();
			return NULL;
		}
		*tmp_client_id_key = tmp_id;
		ne_someip_client_id_info_t* client_id_value = (ne_someip_client_id_info_t*)malloc(sizeof(ne_someip_client_id_info_t));
		if (NULL == client_id_value) {
			ne_someip_log_error("malloc error");
			free(tmp_client_id_key);
			ne_someip_daemon_deinit();
			return NULL;
		}
		client_id_value->is_used = false;
		memset(&client_id_value->unix_addr, 0, sizeof(ne_someip_endpoint_unix_addr_t));

		ne_someip_map_insert(g_daemon->client_id_map, tmp_client_id_key, client_id_value);
		++tmp_id;
	}

    // init reuse rpc
	ne_someip_error_code_t res = ne_someip_common_reuse_manager_init();
	if (ne_someip_error_code_ok != res) {
        ne_someip_log_error("reuse rpc manager init fail.");
        ne_someip_daemon_deinit();
        return NULL;
	}

	return g_daemon;
}

void ne_someip_daemon_deinit()
{
	if (NULL == g_daemon) {
		return;
	}
	ne_someip_sd_deinit();

    ne_someip_common_reuse_manager_deinit();

	if (NULL != g_daemon->unix_endpoint) {
		ne_someip_endpoint_unregister_callback(g_daemon->unix_endpoint);
		ne_someip_endpoint_unix_unref(g_daemon->unix_endpoint);
		g_daemon->unix_endpoint = NULL;
	}
	
	if (NULL != g_daemon->io_looper) {
		ne_someip_looper_unref(g_daemon->io_looper);
		ne_someip_looper_quit(g_daemon->io_looper);
		g_daemon->io_looper = NULL;
	}

	if (NULL != g_daemon->io_thread) {
		ne_someip_thread_unref(g_daemon->io_thread);
		ne_someip_thread_stop(g_daemon->io_thread);
		g_daemon->io_thread = NULL;
	}

	if (NULL != g_daemon->client_id_map) {
		ne_someip_map_unref(g_daemon->client_id_map);
		g_daemon->client_id_map = NULL;
	}

	free(g_daemon);
	g_daemon = NULL;
}

ne_someip_endpoint_unix_t* ne_someip_daemon_get_unix_endpoint()
{
	if (NULL == g_daemon) {
		ne_someip_log_error("daemon is null");
		return NULL;
	}

	return g_daemon->unix_endpoint;
}

bool ne_someip_daemon_get_client_id(const ne_someip_endpoint_unix_addr_t* unix_addr,
	ne_someip_client_id_t* client_id, ne_someip_client_id_t* client_id_min, ne_someip_client_id_t* client_id_max)
{
	ne_someip_log_debug("start");
	if (NULL == g_daemon || NULL == unix_addr || NULL == client_id || NULL == client_id_min
		|| NULL == client_id_max) {
		ne_someip_log_error("g_daemon or unix_addr or client id is null");
		return false;
	}

	*client_id_min = 0;
	*client_id_max = 0;
	*client_id = -1;
	ne_someip_map_iter_t* it = ne_someip_map_iter_new(g_daemon->client_id_map);
	void* key;
	void* value;
	while (ne_someip_map_iter_next(it, &key, &value)) {
		ne_someip_client_id_info_t* client_id_info = (ne_someip_client_id_info_t*)value;
		ne_someip_client_id_t* tmp_client_id = (ne_someip_client_id_t*)key;
		if (NULL == client_id_info) {
			ne_someip_log_error("client info is null");
			return false;
		}

		if (!client_id_info->is_used) {
			strcpy(client_id_info->unix_addr.unix_path, unix_addr->unix_path);
			client_id_info->is_used = true;
			*client_id = *tmp_client_id;
			*client_id_min = NE_SOMEIP_CLIENT_ID_MIN;
			*client_id_max = NE_SOMEIP_CLIENT_ID_MAX;
			break;
		}
	}
	ne_someip_map_iter_destroy(it);

	if (*client_id_min == *client_id_max) {
		return false;
	}
	return true;
}

void ne_someip_daemon_refresh_client_id(const ne_someip_endpoint_unix_addr_t* unix_addr)
{
	if (NULL == g_daemon || NULL == unix_addr) {
		ne_someip_log_error("g_daemon or unix_addr is null");
		return;
	}

	ne_someip_map_iter_t* it = ne_someip_map_iter_new(g_daemon->client_id_map);
	void* key;
	void* value;
	while (ne_someip_map_iter_next(it, &key, &value)) {
		ne_someip_client_id_info_t* client_id_info = (ne_someip_client_id_info_t*)value;
		if (NULL == client_id_info) {
			ne_someip_log_error("client info is null");
			return;
		}

		if (client_id_info->is_used
			&& 0 == strcmp(unix_addr->unix_path, client_id_info->unix_addr.unix_path)) {
			memset(&client_id_info->unix_addr, 0, sizeof(ne_someip_endpoint_unix_addr_t));
			client_id_info->is_used = false;
		}
	}
	ne_someip_map_iter_destroy(it);
}

