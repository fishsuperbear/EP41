/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class CanGateway cpp completion
 */

#include "can_gateway.h"
#include <errno.h>
#include <libgen.h>
#include <linux/can/gw.h>
#include <linux/rtnetlink.h>
#include <net/if.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "diag/docan/log/docan_log.h"

namespace hozon {
namespace netaos {
namespace diag {


#define RTCAN_RTA(r)  ((struct rtattr*)(((char*)(r)) + NLMSG_ALIGN(sizeof(struct rtcanmsg))))
#define RTCAN_PAYLOAD(n) NLMSG_PAYLOAD(n,sizeof(struct rtcanmsg))

/* some netlink helpers stolen from iproute2 package */
#define NLMSG_TAIL(nmsg) \
        ((struct rtattr *)(((char *) (nmsg)) + NLMSG_ALIGN((nmsg)->nlmsg_len)))


CanGateway::CanGateway(N_RouteInfo_t routeInfo)
    : route_info_(routeInfo)
{

}

CanGateway::~CanGateway()
{
}

int32_t CanGateway::Init()
{
    FlushRouteRule(route_info_.if_name);
    for (auto itr : route_info_.forward_table) {
        AddRouteRule(route_info_.if_name, itr.gw_canid_tx, itr.forword_if_name, itr.forword_canid_tx);
    }
    return 0;
}

int32_t CanGateway::Start()
{
    stopFlag_ = false;
    return 0;
}

int32_t CanGateway::Stop()
{
    stopFlag_ = true;
    return 0;
}

N_RouteInfo_t CanGateway::GetRouteInfo()
{
    return route_info_;
}

bool CanGateway::IsStop()
{
    return stopFlag_;
}

int32_t CanGateway::addattr_l(struct nlmsghdr *n, int32_t maxlen, int32_t type, const void *data, int32_t alen)
{
    int32_t len = RTA_LENGTH(alen);
    struct rtattr *rta;

    if ((int32_t)(NLMSG_ALIGN(n->nlmsg_len) + RTA_ALIGN(len)) > maxlen) {
        DOCAN_LOG_E("addattr_l: message exceeded bound of %d", maxlen);
        return -1;
    }
    rta = NLMSG_TAIL(n);
    rta->rta_type = type;
    rta->rta_len = len;
    memcpy(RTA_DATA(rta), data, alen);
    n->nlmsg_len = NLMSG_ALIGN(n->nlmsg_len) + RTA_ALIGN(len);
    return 0;
}


int32_t CanGateway::AddRouteRule(const std::string& fromDevice, const uint16_t fromCanid, const std::string& forwardDevice, const uint16_t fowardCanid)
{
    DOCAN_LOG_D("AddRouteRule cangw -A -s %s -d %s -f %X:7FF -m SET:I:%X.8.0000000000000000", fromDevice.c_str(), forwardDevice.c_str(), fromCanid, fowardCanid);
    uint32_t src_ifindex = if_nametoindex(fromDevice.c_str());
    uint32_t dst_ifindex = if_nametoindex(forwardDevice.c_str());

    struct can_filter filter;
    struct sockaddr_nl nladdr;
    filter.can_id = fromCanid;
    filter.can_mask = 0x7FF;

    int32_t sock= socket(PF_NETLINK, SOCK_RAW, NETLINK_ROUTE);

    struct {
        struct nlmsghdr nh;
        struct rtcanmsg rtcan;
        char buf[1500];
    } req;

    // Add can route rule
    req.nh.nlmsg_flags = NLM_F_REQUEST | NLM_F_ACK;
    req.nh.nlmsg_type  = RTM_NEWROUTE;

    req.nh.nlmsg_len   = NLMSG_LENGTH(sizeof(struct rtcanmsg));
    req.nh.nlmsg_seq   = 0;

    req.rtcan.can_family  = AF_CAN;
    req.rtcan.gwtype = CGW_TYPE_CAN_CAN;
    req.rtcan.flags = 0;

    addattr_l(&req.nh, sizeof(req), CGW_SRC_IF, &src_ifindex, sizeof(src_ifindex));
    addattr_l(&req.nh, sizeof(req), CGW_DST_IF, &dst_ifindex, sizeof(dst_ifindex));
    addattr_l(&req.nh, sizeof(req), CGW_FILTER, &filter, sizeof(filter));


    struct modattr {
        struct can_frame cf;
        __u8 modtype;
        __u8 instruction;
    } __attribute__((packed));

    struct modattr modmsg;
    modmsg.modtype = CGW_MOD_ID;
    modmsg.instruction = CGW_MOD_SET;
    modmsg.cf.can_id = fowardCanid;
    addattr_l(&req.nh, sizeof(req), CGW_MOD_SET, &modmsg, CGW_MODATTR_LEN);

    memset(&nladdr, 0, sizeof(nladdr));
    nladdr.nl_family = AF_NETLINK;
    nladdr.nl_pid    = 0;
    nladdr.nl_groups = 0;

    int32_t err = sendto(sock, &req, req.nh.nlmsg_len, 0,
                (struct sockaddr*)&nladdr, sizeof(nladdr));
    if (err < 0) {
        DOCAN_LOG_E("netlink sendto");
        return err;
    }

    /* clean netlink receive buffer */
    uint8_t rxbuf[8192] = { 0 };
    memset(rxbuf, 0x0, sizeof(rxbuf));
    err = recv(sock, &rxbuf, sizeof(rxbuf), 0);
    if (err < 0) {
        DOCAN_LOG_E("netlink recv");
        return err;
    }
    struct nlmsghdr *nlh = (struct nlmsghdr *)rxbuf;
    if (nlh->nlmsg_type != NLMSG_ERROR) {
        DOCAN_LOG_E("unexpected netlink answer of type %d", nlh->nlmsg_type);
        return -EINVAL;
    }
    struct nlmsgerr *rte = (struct nlmsgerr *)NLMSG_DATA(nlh);
    err = rte->error;
    if (err < 0) {
        DOCAN_LOG_E("netlink error %d (%s)", err, strerror(abs(err)));
    }
    m_sock = sock;
    close(sock);
    m_sock = -1;
    return err;
}

int32_t CanGateway::DelRouteRule(const std::string& fromDevice, const uint16_t fromCanid, const std::string& forwardDevice, const uint16_t fowardCanid)
{
    DOCAN_LOG_D("DelRouteRule cangw -D -s %s -d %s -f %X:7FF -m SET:I:%X.8.0000000000000000", fromDevice.c_str(), forwardDevice.c_str(), fromCanid, fowardCanid);
    uint32_t src_ifindex = if_nametoindex(fromDevice.c_str());
    uint32_t dst_ifindex = if_nametoindex(forwardDevice.c_str());

    struct can_filter filter;
    struct sockaddr_nl nladdr;
    filter.can_id = fromCanid;
    filter.can_mask = 0x7FF;

    int32_t sock= socket(PF_NETLINK, SOCK_RAW, NETLINK_ROUTE);

    struct {
        struct nlmsghdr nh;
        struct rtcanmsg rtcan;
        char buf[1500];
    } req;

    // delete can route rule
    req.nh.nlmsg_flags = NLM_F_REQUEST | NLM_F_ACK;
    req.nh.nlmsg_type  = RTM_DELROUTE;

    req.nh.nlmsg_len   = NLMSG_LENGTH(sizeof(struct rtcanmsg));
    req.nh.nlmsg_seq   = 0;

    req.rtcan.can_family  = AF_CAN;
    req.rtcan.gwtype = CGW_TYPE_CAN_CAN;
    req.rtcan.flags = 0;

    addattr_l(&req.nh, sizeof(req), CGW_SRC_IF, &src_ifindex, sizeof(src_ifindex));
    addattr_l(&req.nh, sizeof(req), CGW_DST_IF, &dst_ifindex, sizeof(dst_ifindex));
    addattr_l(&req.nh, sizeof(req), CGW_FILTER, &filter, sizeof(filter));

    struct modattr {
        struct can_frame cf;
        __u8 modtype;
        __u8 instruction;
    } __attribute__((packed));

    struct modattr modmsg;
    modmsg.modtype = CGW_MOD_ID;
    modmsg.instruction = CGW_MOD_SET;
    modmsg.cf.can_id = fowardCanid;
    addattr_l(&req.nh, sizeof(req), CGW_MOD_SET, &modmsg, CGW_MODATTR_LEN);

    memset(&nladdr, 0, sizeof(nladdr));
    nladdr.nl_family = AF_NETLINK;
    nladdr.nl_pid    = 0;
    nladdr.nl_groups = 0;

    int32_t err = sendto(sock, &req, req.nh.nlmsg_len, 0,
                (struct sockaddr*)&nladdr, sizeof(nladdr));
    if (err < 0) {
        DOCAN_LOG_E("netlink sendto");
        return err;
    }

    /* clean netlink receive buffer */
    uint8_t rxbuf[8192] = { 0 };
    memset(rxbuf, 0x0, sizeof(rxbuf));
    err = recv(sock, &rxbuf, sizeof(rxbuf), 0);
    if (err < 0) {
        DOCAN_LOG_E("netlink recv");
        return err;
    }
    struct nlmsghdr *nlh = (struct nlmsghdr *)rxbuf;
    if (nlh->nlmsg_type != NLMSG_ERROR) {
        DOCAN_LOG_E("unexpected netlink answer of type %d", nlh->nlmsg_type);
        return -EINVAL;
    }
    struct nlmsgerr *rte = (struct nlmsgerr *)NLMSG_DATA(nlh);
    err = rte->error;
    if (err < 0) {
        DOCAN_LOG_E("netlink error %d (%s)", err, strerror(abs(err)));
    }
    m_sock = sock;
    close(sock);
    m_sock = -1;
    return err;
}

int32_t CanGateway::FlushRouteRule(const std::string& fromDevice)
{
    DOCAN_LOG_D("FlushRouteRule cangw -F");
    uint32_t src_ifindex = 0;
    uint32_t dst_ifindex = 0;

    struct sockaddr_nl nladdr;
    int32_t sock= socket(PF_NETLINK, SOCK_RAW, NETLINK_ROUTE);

    struct {
        struct nlmsghdr nh;
        struct rtcanmsg rtcan;
        char buf[1500];
    } req;

    // delete can route rule
    req.nh.nlmsg_flags = NLM_F_REQUEST | NLM_F_ACK;
    req.nh.nlmsg_type  = RTM_DELROUTE;

    /* if_index set to 0 => remove all entries */
    src_ifindex  = 0;
    dst_ifindex  = 0;

    req.nh.nlmsg_len   = NLMSG_LENGTH(sizeof(struct rtcanmsg));
    req.nh.nlmsg_seq   = 0;

    req.rtcan.can_family  = AF_CAN;
    req.rtcan.gwtype = CGW_TYPE_CAN_CAN;
    req.rtcan.flags = 0;

    addattr_l(&req.nh, sizeof(req), CGW_SRC_IF, &src_ifindex, sizeof(src_ifindex));
    addattr_l(&req.nh, sizeof(req), CGW_DST_IF, &dst_ifindex, sizeof(dst_ifindex));

    memset(&nladdr, 0, sizeof(nladdr));
    nladdr.nl_family = AF_NETLINK;
    nladdr.nl_pid    = 0;
    nladdr.nl_groups = 0;

    int32_t err = sendto(sock, &req, req.nh.nlmsg_len, 0,
                (struct sockaddr*)&nladdr, sizeof(nladdr));
    if (err < 0) {
        DOCAN_LOG_E("netlink sendto");
        return err;
    }

    /* clean netlink receive buffer */
    uint8_t rxbuf[8192] = { 0 };
    memset(rxbuf, 0x0, sizeof(rxbuf));
    err = recv(sock, &rxbuf, sizeof(rxbuf), 0);
    if (err < 0) {
        DOCAN_LOG_E("netlink recv");
        return err;
    }
    struct nlmsghdr *nlh = (struct nlmsghdr *)rxbuf;
    if (nlh->nlmsg_type != NLMSG_ERROR) {
        DOCAN_LOG_E("unexpected netlink answer of type %d", nlh->nlmsg_type);
        return -EINVAL;
    }
    struct nlmsgerr *rte = (struct nlmsgerr *)NLMSG_DATA(nlh);
    err = rte->error;
    if (err < 0) {
        DOCAN_LOG_E("netlink error %d (%s)", err, strerror(abs(err)));
    }
    m_sock = sock;
    close(sock);
    m_sock = -1;
    return err;
}





} // end of diag
} // end of netaos
} // end of hozon

/* EOF */