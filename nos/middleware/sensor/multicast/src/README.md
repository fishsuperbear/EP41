/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

---
# nvsipl_multicast sample
The sample demostrates how to implement sending the output of live cameras to multiple consumers via NvStreams.
So far, the consumers include CUDA, encoder and display.

## Usage example
#### Usage information
detailed usage information
`./nvsipl_multicast -h`

#### Intra-process
1. Start SIPL producer, CUDA consumer and encoder consumer in a single process
`./nvsipl_multicast`
2. Show the current version
`./nvsipl_multicast -V`
3. Dump .yuv and .h264 files
`./nvsipl_multicast -f`
4. Process every 2nd frame
`./nvsipl_multicast -k 2`
5. Run for 5 seconds
`./nvsipl_multicast -r 5`
6. List available platform configurations
`./nvsipl_multicast -l`
7. Run with a dynamic platform configuration on non-safety OS, e.g.
`./nvsipl_multicast -g F008A120RM0AV2_CPHY_x4 -m "1 0 0 0"`
8. Specify a static platform configuration, e.g.
`./nvsipl_multicast -t F008A120RM0AV2_CPHY_x4`
9. Enable camera stitching and display. (default display resolution:1920x1080)
Warning: this display example doesn't restrict the number of cameras, whose outputs are to be stitched, but you may hit performance issue as more cameras are involved.
`./nvsipl_multicast -d`
10. Enable multiple ISP outputs(ISP0/ISP1) and use multiple elements to carry them to consumers.
Note: here is the typical scenario:
ISP0 YUV420sp BL -> Encode consumer
ISP1 YUV420sp PL -> Cuda consumer
`./nvsipl_multicast -e`

#### Inter-process (P2P)
1. Default usecase
1.1 Start SIPL producer process
`./nvsipl_multicast -p`
1.2 Start CUDA consumer process
`./nvsipl_multicast -c "cuda"`
1.3 Start encoder consumer process
`./nvsipl_multicast -c "enc"`
2. Run with a dynamic platform configuration on non-safety OS.
Note, please ensure the consistency of platform configuration in between producer and consumers. The peer validation feature will check whether the input command information is consistent with the producer process in each consumer process, and exit the current consumer process if inconsistent, e.g.
`./nvsipl_multicast -g F008A120RM0AV2_CPHY_x4 -m "1 0 0 0" -p`
`./nvsipl_multicast -g F008A120RM0AV2_CPHY_x4 -m "1 0 0 0" -c "cuda"`
`./nvsipl_multicast -g F008A120RM0AV2_CPHY_x4 -m "1 0 0 0" -c "enc"`
3. Enable multiple ISP outputs(ISP0/ISP1)
`./nvsipl_multicast -p -e`
`./nvsipl_multicast -c "cuda" -e`
`./nvsipl_multicast -c "enc" -e`

#### Inter-chip (C2C)
Note: for simplicity, the name of C2C src channels and dst channels are hard-coded.
src channel: nvscic2c_pcie_s1_c5_<n>, n = sensor_index * NUM_IPC_CONSUMERS + consumer_index.
dst channel: nvscic2c_pcie_s2_c5_<n>, n = sensor_index * NUM_IPC_CONSUMERS + consumer_index.
Anyway, you may customize it accordingly.
1. Default usecase
1.1 Start SIPL producer process
`./nvsipl_multicast -P`
1.2 Start CUDA consumer process
`./nvsipl_multicast -C "cuda"`
1.3 Start encoder consumer process
`./nvsipl_multicast -C "enc"`

#### Late-/re-attach (Only for Linux or QNX standard OS)
1. Producer works with encoder consumer at first, then CUDA consumer attach to pipeline lately.
1.1 Start Producer process:
`./nvsipl_multicast -g F008A120RM0A_CPHY_x4 -m "0x0001 0 0 0" -p --late-attach`
1.2 Start Encoder process as an early consumer:
`./nvsipl_multicast -g F008A120RM0A_CPHY_x4 -m "0x0001 0 0 0" -c "enc" --late-attach`
1.3 Start CUDA process as a late consumer:
`./nvsipl_multicast -g F008A120RM0A_CPHY_x4 -m "0x0001 0 0 0" -c "cuda" --late-attach`
1.4 Input 'at' command in the producer process to attach CUDA consumer to pipeline
If all goes well, you will see "Late consumer is attached successfully!" is printed.
1.5 Input 'de' command in the producer process to detach CUDA consumer from pipeline
If all goes well, you will see "Late consumer is detached successfully!" is printed.