#!/bin/bash

interrupts=$(cat /proc/interrupts | grep "b950000.tegra-hsp")

interrupt_number=$(echo "$interrupts" | awk '{print $1}' | tr -d ':')

echo 11 > /proc/irq/$interrupt_number/smp_affinity_list