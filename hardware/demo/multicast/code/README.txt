nvsipl_multicast Sample App - README
Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
NVIDIA Corporation and its licensors retain all intellectual property and proprietary rights in and to this software, related documentation and any modifications thereto. Any use, reproduction, disclosure or distribution of this software and related documentation without an express license agreement from NVIDIA Corporation is strictly prohibited.

---
In nvsipl_multicast sample, there is one NvMedia producer and two consumers (CUDA consumer + encoder consumer).
<V1.0>
1. Support both single process and IPC scenarios.
2. Support multiple cameras
3. Support dumping bitstreams to h264 file on encoder consumer side.
4. Support dumping frames to YUV files on CUDA consumer side.

<V1.1>
1. Support skip specific frames on each consumer side.
2. Add NvMediaImageGetStatus to wait ISP processing done in main to fix the green stripe issue.

<V1.2>
1. Replace NvMediaImageGetStatus with CPU wait to fix the green stripe issue.
2. Add cpu wait before dumping images or bitstream.
3. Support carry meta data to each consumer.
   - Currently, only frameCaptureTSC is included in the meta data.
4. Perform CPU wait after producer receives PacketReady event to WAR the issue of failing to register sync object with ISP.

<V1.3>
1. Improve the H264 encoding quality
2. Code clean.
3. Change fopen mode to "wb"

<V2.0>
1. Migration to the new NvSciBuf* based APIs
2. Add "-f" (file dump), "-k" (frame mod), "-q" (queue type") options
Note, 
v2.0 sample runs well from 6.0.3.1 release.
But on 6.0.3.0, you need to apply some patch .so files beforehand, in order to run the v2.0 sample.

<V2.1.0>
Work on Drive OS 6.0.4.0 and 6.0.5.0 releases.

1. Fix memory leak issue in encoder consumer.
2. Fix warning print of empty fence on qnx
3. Remove deprecated SF3324 support
4. Add version info
5. Add SIPLQuery in non-safety build
6. Add the default NITO-file path
7. Add run for xxx seconds duration
8. Add “-l” option to list available configurations

Please note, you need to prepare a platform configuration header file and put it under the platform directory.

[Usage example]
./nvsipl_multicast -h (detailed usage information)

Intra-process：
1) Start SIPL producer, CUDA consumer and encoder consumer in a single process
   ./nvsipl_multicast
2) Show the current version
   ./nvsipl_multicast -V
3) Dump .yuv and .h264 files
   ./nvsipl_multicast -f
4) Process every 2nd frame
   ./nvsipl_multicast -k 2
5) Run for 5 seconds
   ./nvsipl_multicast -r 5
6) List available platform configurations
   ./nvsipl_multicast -l
7) Run with a dynamic platform configuration on Linux OS.
   ./nvsipl_multicast -g F008A120RM0A_CPHY_x4 -m "1 0 0 0"
8) Specify a static platform configuration
   ./nvsipl_multicast -t F008A120RM0A_CPHY_x4

Inter-process:
9) Start SIPL producer process
   ./nvsipl_multicast -p
10) Start CUDA consumer process
   ./nvsipl_multicast -c “cuda”
11) Start encoder consumer process
   ./nvsipl_multicast -c “enc”
