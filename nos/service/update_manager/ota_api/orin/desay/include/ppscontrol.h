/*
 * ppscontrol.h
 *
 *  Created on: Jun 21, 2022/11/18
 *      Author: uidq3210
 * Copyright Statement
 * libpps_com.so https://github.com/jeremyko/disruptorCpp-IPC-Arbitrary-Length-Data  
 * MIT LICENSE  https://github.com/jeremyko/disruptorCpp-IPC-Arbitrary-Length-Data/blob/master/LICENSE
 */

#ifndef PPSCONTROL_H_
#define PPSCONTROL_H_

#include <functional>

#ifndef USING_DEY_NAMESPACE
	#define USING_DEY_NAMESPACE using namespace DESY;
#endif

//#define	TOPIC_NAME_SIZE		32
#define	TOPIC_NAME_SIZE		255

namespace DESY {

typedef	enum MSGTYPE_DEFINE
{
	MSGTYPE_BEST_EFFORT		= 0,
	MSGTYPE_RELIABLE		= 1
}MSGTYPE_DEFINE_ENUM;

typedef	enum INDEPENDENT_DEFINE
{
	USE_SHARED_THREAD		= 0,
	USE_INDEPENDENT_THREAD	= 1
}INDEPENDENT_DEFINE_ENUM;

typedef	enum SUBSCRIBER_DEFINE
{
	SUBS_DISABLE		= 0,
	SUBS_OPENREAD		= 1,
	SUBS_OPENTRUNC		= 2
}SUBSCRIBER_DEFINE_ENUM;

typedef	enum ON_STATE_DEFINE
{
	ONSTATE_CLEAR_SUBSCRIBE	= 0x10,
}ON_STATE_DEFINE_ENUM;

typedef	enum INIT_STATE_DEFINE
{
	STATE_INIT_NONE			        = 0,
	STATE_INIT_OK			        = 1,
	STATE_INIT_PARAM_ERR	        = 2,
	STATE_INIT_ALOC_MEMORY_ERR		= 3,
	STATE_INIT_SUBSCRIBE_ERR		= 4,
	STATE_INIT_ALOC_THREAD_ERR		= 5
}INIT_STATE_ENUM;

#define	MSG_BESTEFFORT		0x1
#define	MSG_RELIABLE		0x2
#define	MSG_SLEEPWAIT		0x4
#define	MSG_BLOCKWAIT		0x8
#define	MSG_SLEEPMS			0xFF00

typedef void (*message_callback_fun)(int topicid,int cmdid,int size,char *pdata);
typedef void (*on_state_fun)(int topicid,int param,int errid,char *errstr);

typedef std::function<void (int topicid,int cmdid,int size,char *pdata)> message_callback_std;
typedef std::function<void (int topicid,int param,int errid,char *errstr)> on_state_std;

typedef struct PPS_CFG //base support.
{
	int				topicid;
	char 			topicname[TOPIC_NAME_SIZE];
	int				bufcnt;//max buffer cnt of one cmd.
	int				issub;// is sub?
	int				ispub;// is pub?
	int				cmdcnt;//max cmd count
	int				datasize;//max cmd data size;
}PPS_CFG_STRU;

typedef struct PPS_CFG_EX //support independent_thread and priority.
{
	int				topicid;
	char 			topicname[TOPIC_NAME_SIZE];
	int				bufcnt;//max buffer cnt of one cmd.
	int				issub;// is sub?
	int				ispub;// is pub?
	int				cmdcnt;//max cmd count
	int				datasize;//max cmd data size;
	int				independent_thread;
	int				priority;

}PPS_CFG_EX_STRU;

typedef struct PPS_CFG_EXX //support function safety msgtype.
{
	int				topicid;
	char 			topicname[TOPIC_NAME_SIZE];
	int				bufcnt;//max buffer cnt of one cmd.
	int				issub;// is sub?
	int				ispub;// is pub?
	int				cmdcnt;//max cmd count
	int				datasize;//max cmd data size;
	int				independent_thread;
	int				priority;
	int				msgtype;//bit0:besteffort msg  bit1:reliable msg bit2:sleepwait bit3:blockwait. bit8~bit15 sleepms
}PPS_CFG_EXX_STRU;

class HalSubInterface {
public:
  virtual void onHalSubInterface(int topicid,int cmdid,int size,char *pdata) = 0;
};

class ppscontrol {
public:
    static ppscontrol* Instance();//just get handle
	static ppscontrol* Instance(PPS_CFG_STRU *ppscfg,int size);//create and get handle
	static ppscontrol* Instance(PPS_CFG_EX_STRU *ppscfg,int size);//create and get handle
	static ppscontrol* Instance(PPS_CFG_EXX_STRU *ppscfg,int size);//create and get handle

	void registerCallback(message_callback_fun pcallback);//register static receive callback.
	void registerCallback(message_callback_std pcallback);//register std:function callback.
	void registerCallback(HalSubInterface* data_receiver);//register interface receive callback.
	bool publish(int topic,int cmdid,int size,char *data);//send message.
	bool subscribe(int topicid,int stat);//0:unsubscribe 1:subscribe.
	//return 1~max success  <0 error.
	int64_t send(int topic,int cmdid,int size,void *data);//send message.

	//add interface 20220521
	void registeronstate(on_state_fun pcallback);//register static receive callback.
	void registeronstate(on_state_std pcallback);//register std:function callback.
	/*pub_force 1:yes 0:no.wait for pub_wait_ms.*/
	bool setstrategy(int pub_force,int pub_wait_ms);
	/*load buffer. return chunk_idx . < 0 :means errorid .*/
	int64_t  loan_chunk(int topicid,int cmdid,void **userPayload,int size);
	/*commit chunk_idx  buffer. 1:success; < 0:means errorid.*/
	int publish_chunk(int topicid,int64_t  chunk_idx);

	static const char * version();
	static ppscontrol* getInstance(void); //create or get a new empty instance.
	int getInstanceState(void);

	int initial(PPS_CFG_EX_STRU  *ppscfg, int size, message_callback_fun pcallback = nullptr);//init topic.
	int initial(PPS_CFG_EXX_STRU *ppscfg, int size, message_callback_fun pcallback = nullptr);//init topic.

	void registerCallback(int topicid,message_callback_fun pcallback);//register static receive callback.
	void registerCallback(int topicid,message_callback_std pcallback);//register std:function callback.
	void registerCallback(int topicid,HalSubInterface* data_receiver);//register interface receive callback.

private:
	static ppscontrol* instance;

    ppscontrol();
    ppscontrol(PPS_CFG_STRU *ppscfg,int size);
    ppscontrol(PPS_CFG_EX_STRU *ppscfg,int size);
    ppscontrol(PPS_CFG_EXX_STRU *ppscfg,int size);

	~ppscontrol();

	class ppscontrolImpl;

private:
    ppscontrolImpl * _pimpl;
};

}//end namesapce

#endif /* PPSCONTROL_H_ */
