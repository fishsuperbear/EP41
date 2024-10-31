# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

include ../../../../make/nvdefs.mk

TARGETS = libnvsipl_multicast.so

CXXFLAGS := $(NV_PLATFORM_OPT) $(NV_PLATFORM_CFLAGS) -I. -I../../../../include/nvmedia_6x/ -I..
CXXFLAGS += -std=c++14 -fexceptions -frtti -fPIC
CPPFLAGS := $(NV_PLATFORM_CPPFLAGS) $(NV_PLATFORM_SDK_INC) $(NV_PLATFORM_CXXFLAGS)
CPPFLAGS += -I./platform -frtti
LDFLAGS := $(NV_PLATFORM_SDK_LIB) $(NV_PLATFORM_TARGET_LIB) $(NV_PLATFORM_LDFLAGS)

ifeq ($(NV_PLATFORM_SAFETY),1)
CPPFLAGS += -DNV_IS_SAFETY=1
endif

OBJS := CPoolManager.o
OBJS += CProducer.o
OBJS += CSIPLProducer.o
OBJS += CConsumer.o
OBJS += CCudaConsumer.o
OBJS += CClientCommon.o
OBJS += CEncConsumer.o
OBJS += CUtils.o
OBJS += CDisplayConsumer.o
OBJS += CStitchingConsumer.o
OBJS += CDisplayProducer.o
OBJS += COpenWFDController.o
OBJS += CCmdLineParser.o
OBJS += CFactory.o
OBJS += CAppConfig.o
OBJS += CPeerValidator.o
OBJS += CLateConsumerHelper.o
OBJS += main.o
OBJS += CEncManager.o

LDLIBS += -lnvsipl
ifeq ($(NV_PLATFORM_SAFETY),0)
LDLIBS += -lnvsipl_query
endif
LDLIBS += -lnvmedia_iep_sci
LDLIBS += -lnvscistream
LDLIBS += -lnvmedia2d
LDLIBS += -lnvscibuf
LDLIBS += -lnvscisync
LDLIBS += -lnvscievent
LDLIBS += -lnvsciipc
LDLIBS += -lnvscicommon
LDLIBS += -lcuda
LDLIBS += -ltegrawfd
ifeq ($(NV_PLATFORM_OS),QNX)
  CPPFLAGS += -DNVMEDIA_QNX
  LDLIBS += $(NV_PLATFORM_CUDA_LIB)/libcudart_static.a
  LDLIBS += -lnvdtcommon
else
  LDLIBS += -L$(NV_PLATFORM_CUDA_TOOLKIT)/targets/aarch64-linux/lib/ -lcudart
  LDLIBS += -lpthread
endif

.PHONY: default
default: $(TARGETS)

$(TARGETS): $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS) -shared

clean clobber:
	rm -rf $(OBJS) $(TARGETS)
