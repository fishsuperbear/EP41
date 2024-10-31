/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

## Release Note

#### V2.5.2
1. store and release the meta buffer to avoid memory leak.

#### V2.5.1
1. fix the query timeout for display consumer.

#### V2.5.0
1. Support late-/re-attach (Only for Linux or QNX standard OS)
Note: Late-/re-attach feature and multi-elements feature can not be enabled together.
2. Support peer validation in p2p communication

#### V2.4.2
1. optimize the stitching&display.

#### V2.4.1
1. Fix EncConsumer bug about dump file, set type to IDR

#### V2.4.0
1. Support chip-to-chip communication
2. Change platform config name from F008A120RM0A_CPHY_x4 to F008A120RM0AV2_CPHY_x4 for AR0820

#### V2.3.1
1. Refactor configuration design
   1.1 Extract CAppConfig class
   1.2 Define CommType, EntityType, ConsumerType, QueueType
2. Refacotr CFactory
   2.1. Change CFactory to be a singleton class
3. Refactor CPoolManager
   3.1 Extract HandleElements(), HandleBuffers() and FreeElements().
4. Add C2C support

#### V2.3.0
1. Using multiple elements to support processing multiple ISP's outputs

#### V2.2.0
1. Add display consumer
   1.1 Stitch output from multiple cameras before sending them to display

#### V2.1.1
1. Fix cudaconsumer cuda copy regression on 6050.
V2.1.1 runs well on Driver OS 6.0.5.0 release. V2.1.0 will face cudaconsumer copy error on 6050.

#### V2.1.0
1. Fix memory leak issue in encoder consumer.
2. Fix warning print of empty fence on qnx
3. Remove deprecated SF3324 support
4. Add version info
5. Add SIPLQuery in non-safety build
6. Add the default NITO-file path
7. Add run for xxx seconds duration
8. Add “-l” option to list available configurations

V2.1.0 runs well on Driver OS 6.0.4.0 release.
Please note, you need to prepare a platform configuration header file and put it under the platform directory.

#### V2.0
1. Migration to the new NvSciBuf* based APIs
2. Add "-f" (file dump), "-k" (frame mod), "-q" (queue type") options
Note,
v2.0 sample runs well from 6.0.3.1 release.
But on 6.0.3.0, you need to apply some patch .so files beforehand, in order to run the v2.0 sample.

#### V1.3
1. Improve the H264 encoding quality
2. Code clean.
3. Change fopen mode to "wb"

#### V1.2
1. Replace NvMediaImageGetStatus with CPU wait to fix the green stripe issue.
2. Add cpu wait before dumping images or bitstream.
3. Support carry meta data to each consumer.
   - Currently, only frameCaptureTSC is included in the meta data.
4. Perform CPU wait after producer receives PacketReady event to WAR the issue of failing to register sync object with ISP.

#### V1.1
1. Support skip specific frames on each consumer side.
2. Add NvMediaImageGetStatus to wait ISP processing done in main to fix the green stripe issue.

#### V1.0
1. Support both single process and IPC scenarios.
2. Support multiple cameras
3. Support dumping bitstreams to h264 file on encoder consumer side.
4. Support dumping frames to YUV files on CUDA consumer side.