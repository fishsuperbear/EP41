/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: xpcshm common header, for both user state and kernel state
 * Author: Huawei Godel Lab
 * Create: 2020-10-10
 */

#ifndef XPCSHM_COMMON_H
#define XPCSHM_COMMON_H

#if defined(LINUX_CORE_SUPP) || defined(ON_CORE)
#define SD_LOCAL_CPU_ID 1U
#define DP_LOCAL_CPU_ID 5U
#else
#define SD_LOCAL_CPU_ID 0x80100
#define DP_LOCAL_CPU_ID 0x90100
#endif
#define REAL_DOMAIN_COUNT 2U

#define XPC_CHANNEL_ID_MAX (512U)
#define XPC_MAX_DOMAIN_NUM (16U + 1U)
#define XPC_CHANNEL_ID_MAX_ALL_DOMAIN (XPC_CHANNEL_ID_MAX * XPC_MAX_DOMAIN_NUM)

#define XPC_CHANNEL_NAME_MAX 32U
#define XPC_DYNA_MAX_SIZE (16U * 1024U * 1024U) // 16MB
#define FIFO_SIZE_MAX (80U * 1024U) // 80KB
#define MAX_PKG_IN_FIFO (FIFO_SIZE_MAX / 2U)
#define FIFO_SIZE_MIN (1) // 1Byte
#define FIFO_ZC_BUF_SIZE_SPLIT (2U * 1024U) // 2K
#define XPC_MMAP_MIN_SIZE ((8U * 1024U) + 1U) // > 8K
#define XPC_MMAP_MAX_SIZE (50U * 1024U * 1024U) // 50M

#define STASTIC_SIZE_FACTOR 2U
#define XPC_MEM_MIN (200U * 1024U * 1024U) // 200M
#define XPC_MEM_MAX (UINT_MAX) // 4G - 1 4U * 1024U * 1024U * 1024U -1U
#define XPC_MEM_STATIC_DYN_SPIT (2U) // static dynamic half of whole size
#define XPC_FIFO_MAX_SIZE (XPC_MEM_MAX / XPC_MEM_STATIC_DYN_SPIT / XPC_CHANNEL_ID_MAX / STASTIC_SIZE_FACTOR)

// multi domain defines
#define FIFO_MULTI_DOMAIN_BUF_SIZE_SPILT FIFO_ZC_BUF_SIZE_SPLIT
#define XPC_MULTI_DOMAIN_QUEUE_PACK_SIZE FIFO_MULTI_DOMAIN_BUF_SIZE_SPILT
#define XPC_MULTI_DOMAIN_QUEUE_PACK_COUNT (79U)
#define XPC_MULTI_DOMAIN_QUEUE_PACK_SHIFT (11)

// 4 * 4K = 2 ^ 14 and this two MACRO must change together
#define XPC_PAGE_SHIFT 14U // must be the nth power of 2
#define XPC_DYN_MEM_PAGE_BLOCK (4U) // (1 page / 4K) change to 4 page one block

#define XPC_4K_PAGE_SHITF (12U) // 4k is 2^12
#define XPC_DYN_MEM_PAGE_BLOCK_MAX (1024U) // max dyna block size is 4M(under 4k page)

#define XPC_PAGE_SIZE (1UL << XPC_PAGE_SHIFT)
#define XPC_PAGE_MASK (~(XPC_PAGE_SIZE - 1UL))

#define XPC_PAGE_ALIGN(n) (((n) + XPC_PAGE_SIZE - 1UL) & XPC_PAGE_MASK)

#ifndef ON_CORE
#define XPCSHM_DEV_PATH "/dev/xpcshm"
#else
#define XPCSHM_DEV_PATH "/local/dev/xpcshm"
#endif

#define IOCTL_CMD_TO_FLAG_SHIFT 16U
#define MAX_POLL_NUM 50U
#define INVALID_CHANNEL_ID (-1)

#define XPC_4K_PAGE_SHIFT (12U) // 4k is 2 ^ 12

// channel error number
#define XPC_ERR_BASE (-65536)
#define XPC_ERR_CHL_NUM_LIMIT (XPC_ERR_BASE + 1) // -65535 通道数超过限制
#define XPC_ERR_CHL_SIZE_LIMIT (XPC_ERR_BASE + 2) // size(fifo size/max pkt size)大小超过上下限
#define XPC_ERR_CHL_NOT_MATCH (XPC_ERR_BASE + 3) // open/close channel 的时候参数不匹配或者读写channel的时候并没有打开此ID
#define XPC_ERR_CHL_ALREADY_OPEN (XPC_ERR_BASE + 4) // 通道已存在，不能重复打开
#define XPC_ERR_CHL_NOT_EXIST (XPC_ERR_BASE + 5) // 通道不存在或者ID超过上下限
#define XPC_ERR_CHL_CLOSE (XPC_ERR_BASE + 6) // -65530 通道已经关闭
#define XPC_ERR_CHL_TIMEOUT (XPC_ERR_BASE + 7) // 通道操作超时
#define XPC_ERR_CHL_FULL (XPC_ERR_BASE + 8) // 通道已满 无法继续写入
#define XPC_ERR_CHL_EMPTY (XPC_ERR_BASE + 9) // 通道空 无法继续读出
#define XPC_ERR_PKT_OUTSIZE (XPC_ERR_BASE + 10) // 包过大
#define XPC_ERR_ALLOC_OUTMEM (XPC_ERR_BASE + 11) // -65525 动态分配区已满
#define XPC_ERR_MEM_FULL (XPC_ERR_BASE + 12) // 共享內存已滿
#define XPC_ERR_DOMAIN (XPC_ERR_BASE + 13) // 域指定错误
#define XPC_ERR_PARAM (XPC_ERR_BASE + 14) // 参数不对
#define XPC_ERR_GENERAL (XPC_ERR_BASE + 15) // 通用错误(malloc/memset/copy)etc
#define XPC_SHM_DEVICE_NOT_OPEN (XPC_ERR_BASE + 16) // -65520 设备节点没有打开
#define XPC_ERR_PKT_TYPE_NOT_MATCH (XPC_ERR_BASE + 17) // 包类型不对
#define XPC_ERR_SHLIST_OPER (XPC_ERR_BASE + 18) // 操作shlist相关的错误
#define XPC_ERR_PROC_CHL_NOT_EXIT (XPC_ERR_BASE + 19) // 进程信息未被初始化
#define XPC_ERR_PROC_CHL_REOPEN (XPC_ERR_BASE + 20) // 重复打开相同的通道
#define XPC_ERR_PROC_CHL_LAPPED (XPC_ERR_BASE + 21) // -65515 读的慢的进程被套圈,但下次可继续读（重设读指针）
#define XPC_ERR_MUST_USE_DYNAMIC (XPC_ERR_BASE + 22) // 请使用动态buffer
#define XPC_ERR_USER_BUFF_SMALL (XPC_ERR_BASE + 23) // 用户侧buff太小
#define XPC_ERR_BUFF_ALREADY_SEND (XPC_ERR_BASE + 24) // 0拷贝buffer已经发送
#define XPC_ERR_BUFF_ALREADY_FREE (XPC_ERR_BASE + 25) // 多播buffer为空
#define XPC_ERR_IOCTL_CMD_UNKNOW (XPC_ERR_BASE + 26) // -65510 未知的IOCTL CMD
#define XPC_ERR_GET_VMA (XPC_ERR_BASE + 27) // 获取vma失败
#define XPC_ERR_REMAP (XPC_ERR_BASE + 28) // REMAP物理页失败
#define XPC_ERR_GET_PAGE_INFO (XPC_ERR_BASE + 29) // 获取page相关信息错误
#define XPC_ERR_ALREADY_EXIST (XPC_ERR_BASE + 30) // 创建时指定 CHL_O_EXECL 已存在则报此错
#define XPC_ERR_POLL_ITEM_NOT_FOUND (XPC_ERR_BASE + 31) // -65505 poll item is not in the poll list
#define XPC_ERR_POLL_WAIT_EVENT (XPC_ERR_BASE + 32) // poll wait event fail
#define XPC_ERR_POLL_TIMEOUT (XPC_ERR_BASE + 33) // poll time out
#define XPC_ERR_KEY_NOT_EXIST (XPC_ERR_BASE + 34) // 指定的key值不存在
#define XPC_ERR_CHL_NOT_OPEN (XPC_ERR_BASE + 35) // 通道非双/多域打开
#define XPC_ERR_KEY_DUPLICATE (XPC_ERR_BASE + 36) // -65500 创建时key重复
#define XPC_ERR_INVALID_PKT_INFO (XPC_ERR_BASE + 37) // 无效的包类型
#define XPC_ERR_ALLOC_FAILED (XPC_ERR_BASE + 38) // Alloc 失败
#define XPC_ERR_WRONG_READ_METHOD (XPC_ERR_BASE + 39)
#define XPC_ERR_READ_OPS_ON_NAMED_BLK (XPC_ERR_BASE + 40) // 读端操作了有名页（只能写端操作）
#define XPC_ERR_NAMED_BLK_OPS_NOT_PERMIT (XPC_ERR_BASE + 41) // -65495 进行的操作不属于有名页状态，例如IDLE状态下想要释放
#define XPC_ERR_UNKNOW_BLK_CMD (XPC_ERR_BASE + 42)
#define XPC_ERR_SET_NAMED_BLK_STATE (XPC_ERR_BASE + 43)
#define XPC_ERR_NAMED_BLK_NOT_MATCH (XPC_ERR_BASE + 44) // 读写两端的named blk信息不匹配
#define XPC_ERR_INVALID_NAMED_BLK_INFO (XPC_ERR_BASE + 45) // named blk的信息非法
#define XPC_ERR_NAMED_BLK_NOT_SYNC (XPC_ERR_BASE + 46) // -65490 操作需要等待named blk同步之后才能处理
#define XPC_ERR_ZAP_VMA (XPC_ERR_BASE + 47)
#define XPC_ERR_WRITE_CREATE_NAMED_BLK (XPC_ERR_BASE + 48) // 写端创建named blk失败
#define XPC_ERR_MAP_ZERO_PAGE (XPC_ERR_BASE + 49)
#define XPC_ERR_MULTICAST_MAP (XPC_ERR_BASE + 50)
#define XPC_ERR_MULTI_OPEN_ON_MMAPPED_CHL (XPC_ERR_BASE + 51) // -65485
#define XPC_ERR_MMAPED_ON_OTHER_CHL (XPC_ERR_BASE + 52)
#define XPC_ERR_UNMAPP_NAMED_BLK (XPC_ERR_BASE + 53)
#define XPC_ERR_MMAP_OUT_MAX_NUM (XPC_ERR_BASE + 54) // 试图mmap超过MAX_NUM的named blk
#define XPC_ERR_NAMED_BLK_NOT_OWNED (XPC_ERR_BASE + 55) // 试图unmap不属于自己的named blk
#define XPC_ERR_READ_CREATE_NAMED_BLK (XPC_ERR_BASE + 56) // // -65480 试图unmap不属于自己的named blk
#define XPC_ERR_FIFO_READ_NEXT_TYPE_SYNC (XPC_ERR_BASE + 57) // next type is sync pkt when doing fifo_read
#define XPC_ERR_ALLOC_ID_NOT_MATCH (XPC_ERR_BASE + 58)
#define XPC_ERR_INVALID_BLK_CREATE_SYNC_MSG (XPC_ERR_BASE + 59)
#define XPC_ERR_INVALID_CHANNEL_TYPE (XPC_ERR_BASE + 60) // 通道类型不匹配
#define XPC_ERR_MULTI_DOMAIN_WRITE_LEN (XPC_ERR_BASE + 61) // -65475 多域多播写长度超限制
#define XPC_ERR_MULTI_CHL_NOT_SUPPORT_MMAP (XPC_ERR_BASE + 62) // 多域多播不支持Mmap
#define XPC_ERR_INVALID_PKT_ON_CORE (XPC_ERR_BASE + 63)
#define XPC_ERR_VMA_INFO (XPC_ERR_BASE + 64)
#define XPC_ERR_VMA_DEL_FAIL (XPC_ERR_BASE + 65)
#define XPC_ERR_VMA_TO_FREE (XPC_ERR_BASE + 66)
#define XPC_ERR_VMA_ROOT_DESTROYING (XPC_ERR_BASE + 67)
#define XPC_ERR_VMA_NOT_EXIST (XPC_ERR_BASE + 68)
// channel error number end

// channel open flag
#define CHL_O_CREATE 0x0001U // 不存在指定Channel则创建，存在则打开
#define CHL_O_EXECL 0x0002U // 创建channel，存在则报错 XPC_ERR_ALREADY_EXIST
#define CHL_O_OPENONLY 0x0004U // 检查Channel是否已经创建，已创建则返回ChannelID，否则返回错误
#define CHL_O_BYKEY 0x0008U // 以KEY的方式打开 channel, 前提是已存在
#define CHL_O_MULTICAST_CHL 0x0010U // 多域多播形式打开通道，可与上面的标志叠加
// channel open flag end

// channel state
#define XPC_CHL_STATE_CLOSE 0x1000 // 通道关闭
#define XPC_CHL_STATE_OPENING 0x1001 // 单域打开
#define XPC_CHL_STATE_OPEN 0x1002 // 双/多域打开
#define XPC_CHL_STATE_RESET 0x1003 // 以下预留
#define XPC_CHL_STATE_ERROR 0x1004
// channel state end

// package type
#define PKT_TYPE_NORMAL 0x0001U // 一个完整的普通包，包含数据实体
#define PKT_TYPE_PTR 0x0002U // 指针类型包，包内携带的是数据实体所在的指针
#define PKT_TYPE_STREAM 0x0004U // 备用类型，stream类型包，对应stream类型的Channel时使用
/* Channel支持大包分割时，只允许一个发送者 */
#define PKT_TYPE_BEGIN 0x0008U // 备用类型，用于大包分割传送时的第一个包
#define PKT_TYPE_CONTINUE 0x0010U // 备用类型，用于大包分割传送时的中间包
#define PKT_TYPE_END 0x0020U // 备用类型，用于大包分割传送时的最后个包
#define PKT_TYPE_SWITCH_INNER 0x0040U // xpc write 转为零拷贝 內部类型
#define PKT_TYPE_TO_QUE_INNER 0x0080U // xpc 拷贝包到queue中 内部类型
#define PKT_TYPE_BLK_CREATE_SYNC_INNER 0x0100U
#define PKT_TYPE_BLK_CLOSE_SYNC_INNER 0x0200U
#define PKT_TYPE_BLK_MSG 0x0400U
#define PKT_TYPE_INVALID 0x0800U // 类型边界
// package type end

// real domain bitwise for high 16 bit
#define DOMAIN_SD_BITWISE 0x01000000U // 0000 0001 0...
#define DOMAIN_DP_BITWISE 0x02000000U // 0000 0010 0...
#define DOMAIN_DP_CORE_BITWISE 0x04000000U // 0000 0100 0...
#define DOMAIN_SD_CORE_BITWISE 0x08000000U // 0000 1000 0...
// real domain bitwise end

#define DOMAIN_REAL_MASK 0xFFFF0000U // real domain mask (high 16 bit)
#define DOMAIN_VIRT_MASK 0x0000FFFFU // virtual domain mask (low 16 bit)
#define DOMAIN_REAL_SHIFT 24
#define DOMAIN_TO_INDEX(x) ((x) >> 1)

// real domain for dts and kernel
#define DOMAIN_BASE 0x10U
#define DOMAIN_SD DOMAIN_BASE
#define DOMAIN_DP 0x11U
#define DOMAIN_COM 0x12U
#define DOMAIN_CLUSTER 0x13U
#define DOMAIN_DP_CORE 0x14U
#define DOMAIN_SD_CORE 0x15U
// real domain end

#define DOMAIN_BITMAP_SD 0x00
#define DOMAIN_BITMAP_DP 0x01
#define DOMAIN_BITMAP_SD_CORE 0x02
#define DOMAIN_BITMAO_DP_CORE 0x03
#define DOMAIN_BITMAP_BASE 2

// virtual domain for user (low 16 bit)
#define DOMAIN_AOS_CORE 0x01 // 0000 0001 low 16 bit
// virtual domain end
#define XPCSHM_RESET_MEM_VALUE 0U


// xpc ioctl cmd
#define XPCSHM_IO_CTL_BASE 0x1000U
#define XPCSHM_POLL XPCSHM_IO_CTL_BASE
#define XPCSHM_OPEN (XPCSHM_IO_CTL_BASE + 1U)
#define XPCSHM_CLOSE (XPCSHM_IO_CTL_BASE + 2U)
#define XPCSHM_WRITE (XPCSHM_IO_CTL_BASE + 3U)
#define XPCSHM_READ (XPCSHM_IO_CTL_BASE + 4U)
#define XPCSHM_GET_STATUS (XPCSHM_IO_CTL_BASE + 5U)
#define XPCSHM_GET_PERFORMANCE (XPCSHM_IO_CTL_BASE + 6U)
#define XPCSHM_GET_MEMINFO (XPCSHM_IO_CTL_BASE + 7U)
#define XPCSHM_GET_CHLINFO (XPCSHM_IO_CTL_BASE + 8U)
#define XPCSHM_FREE (XPCSHM_IO_CTL_BASE + 9U)
#define XPCSHM_SEND_BUFFER (XPCSHM_IO_CTL_BASE + 10U)
#define XPCSHM_GET_BUFFER (XPCSHM_IO_CTL_BASE + 11U)
#ifndef ON_CORE
#define XPCSHM_MMAP (XPCSHM_IO_CTL_BASE + 12U)
#define XPCSHM_ALLOC (XPCSHM_IO_CTL_BASE + 13U)
#define XPCSHM_IO_CTL_END (XPCSHM_IO_CTL_BASE + 14U + XPCSHM_RESET_MEM_VALUE) // end value of ioctl
#else
#define XPCSHM_ALLOC (XPCSHM_IO_CTL_BASE + 12U)
#define XPCSHM_IO_CTL_END (XPCSHM_IO_CTL_BASE + 13U + XPCSHM_RESET_MEM_VALUE) // end value of ioctl
#endif // ON_CORE
// xpc ioctl cmd end

#define IOCTL_BULK (XPCSHM_IO_CTL_END - XPCSHM_IO_CTL_BASE)

// named memblk
#define NAMED_BLK_MAX_NUM 2U
#define NAMED_BLK_MAX_NAME_LEN 8U
#define NAMED_BLK_MAX_NAME_ARRAY_LEN (NAMED_BLK_MAX_NAME_LEN + 1U)
#define NAMED_BLK_MSG_NUM 3U

#define NAMED_BLK_STATE_IDLE 0X100U
#define NAMED_BLK_STATE_WRITE_CREATE 0x101U
#define NAMED_BLK_STATE_READ_CREATE 0x102U
#define NAMED_BLK_STATE_SYNCING 0x103U
#define NAMED_BLK_STATE_WORKING 0x104U
#define NAMED_BLK_STATE_READ_CLOSE 0x105U
#define NAMED_BLK_STATE_WRITE_CLOSE_SYNC 0x106U
#define NAMED_BLK_STATE_WAIT_SYNC 0x107U
#define NAMED_BLK_STATE_INVALID_OPS 0x108U

#define XPC_SHM_TYPE_READ 0x100U
#define XPC_SHM_TYPE_WRITE 0x101U

#define CLEAN_NAMED_BLK_FROM_FREE 0x100U
#define CLEAN_NAMED_BLK_FROM_CLOSE_CHL 0x101U
#define CLEAN_NAMED_BLK_FROM_RELEASE 0x102U

#define NAMED_BLK_NUM_POS 0
#define NAMED_BLK_OFFSET_POS 1
#define NAMED_BLK_LEN_POS 2

#define TIME_SWITCH_BASE 1000L

#define XPC_DP_CORE_CPU_ID 4U

#define XPCSHM_NR_CPUS 16U
#define XPCSHM_INVALID_CPUID XPCSHM_NR_CPUS
#define XPCSHM_DEFAULT_CORE_CPUID (XPCSHM_INVALID_CPUID + 1U)

#endif
