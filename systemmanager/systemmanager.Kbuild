###########################################################################
# Kbuild fragment for systemmanager.ko
###########################################################################

#
# Define SYSTEMMANAGER_{SOURCES,OBJECTS}
#

SYSTEMMANAGER_SOURCES =
SYSTEMMANAGER_SOURCES += systemmanager/systemmanager.c
SYSTEMMANAGER_SOURCES += systemmanager/common/neta_common.c
SYSTEMMANAGER_SOURCES += systemmanager/common/neta_klog.c
SYSTEMMANAGER_SOURCES += systemmanager/common/neta_kerr.c
SYSTEMMANAGER_SOURCES += systemmanager/node/neta_node_common.c
SYSTEMMANAGER_SOURCES += systemmanager/node/neta_node.c
SYSTEMMANAGER_SOURCES += systemmanager/node/lognode/neta_log.c
SYSTEMMANAGER_SOURCES += systemmanager/node/lognode/neta_lognode.c
SYSTEMMANAGER_SOURCES += systemmanager/node/lognode/neta_logblock.c
SYSTEMMANAGER_SOURCES += systemmanager/node/lognode/neta_lognode_ioctl_readerimpl.c
SYSTEMMANAGER_SOURCES += systemmanager/node/lognode/neta_lognode_ioctl_writerimpl.c
SYSTEMMANAGER_SOURCES += systemmanager/node/halnode/neta_halnode.c
SYSTEMMANAGER_SOURCES += systemmanager/node/halnode/neta_hal.c
SYSTEMMANAGER_SOURCES += systemmanager/node/halnode/neta_hallog_ctrl.c
SYSTEMMANAGER_SOURCES += systemmanager/node/lognode/neta_group_manager.c

SYSTEMMANAGER_OBJECTS = $(patsubst %.c,%.o,$(SYSTEMMANAGER_SOURCES))

obj-m += systemmanager.o
systemmanager-y := $(SYSTEMMANAGER_OBJECTS)

SYSTEMMANAGER_KO = systemmanager/systemmanager.ko

NV_KERNEL_MODULE_TARGETS += $(SYSTEMMANAGER_KO)

#
# Define systemmanager.ko-specific CFLAGS.
#

SYSTEMMANAGER_CFLAGS += -I$(src)/systemmanager
SYSTEMMANAGER_CFLAGS += -I$(src)/systemmanager/common
SYSTEMMANAGER_CFLAGS += -I$(src)/systemmanager/node
SYSTEMMANAGER_CFLAGS += -I$(src)/systemmanager/node/lognode
SYSTEMMANAGER_CFLAGS += -I$(src)/systemmanager/node/halnode
SYSTEMMANAGER_CFLAGS += -UDEBUG -U_DEBUG -DNDEBUG -DNV_BUILD_MODULE_INSTANCES=0
SYSTEMMANAGER_CFLAGS += -Wno-date-time

$(call ASSIGN_PER_OBJ_CFLAGS, $(SYSTEMMANAGER_OBJECTS), $(SYSTEMMANAGER_CFLAGS))

#
# Register the conftests needed by systemmanager.ko
#

#NV_OBJECTS_DEPEND_ON_CONFTEST += $(SYSTEMMANAGER_OBJECTS)

#NV_CONFTEST_FUNCTION_COMPILE_TESTS += get_user_pages

