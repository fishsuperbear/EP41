#!/bin/sh

init_coredump()
{
    ulimit -c 1024000
    mkdir -p /opt/usr/data/coredump
    echo "/opt/usr/data/coredump/core.%E-%p-%t" > /proc/sys/kernel/core_pattern
    chmod -f 777  /opt/usr/data/coredump
    echo 1 > /proc/sys/fs/suid_dumpable
    chown -RPf nvidia:nvidia  /opt/usr/data/coredump
}


init_coredump


