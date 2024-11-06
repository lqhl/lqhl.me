---
title: 链路聚合
date: 2018-05-15T12:22:34+08:00
tags: ['Linux', 'Networking']
---

链路聚合（bonding or link aggregation）将多个物理网卡捆绑在一起，形成一个逻辑端口。它可以用于增加链路带宽和提供链路的冗余与容错。
以下是我参考 [Ubuntu Bonding](https://help.ubuntu.com/community/UbuntuBonding) 后在 Ubuntu 16.04 上配置的步骤：

1. 安装：

    ```bash
    sudo apt-get install ifenslave
    ```

2. 载入提供 bonding 的 kernel module：

    ```bash
    sudo modprobe bonding
    ```

3. 在电脑启动时自动载入 `bonding` 模块，将 `bonding` 加入 `/etc/modules`：

    ```ini
    # /etc/modules: kernel modules to load at boot time.
    #
    # This file contains the names of kernel modules that should be loaded
    # at boot time, one per line. Lines beginning with "#" are ignored.

    bonding
    ```

4. 配置 `/etc/network/interfaces`：

    ```bash
    # This file describes the network interfaces available on your system
    # and how to activate them. For more information, see interfaces(5).

    source /etc/network/interfaces.d/*

    # The loopback network interface
    auto lo
    iface lo inet loopback

    auto ens6f0
    iface ens6f0 inet manual
        bond-master bond0

    auto ens6f1
    iface ens6f1 inet manual
        bond-master bond0

    auto bond0
    iface bond0 inet static
        address 192.168.10.100
        netmask 255.255.255.0
        bond-mode 6
        bond-miimon 100
        bond-slaves ens6f0 ens6f1
    ```

5. bond-mode 有多重选择，参考 [Ubuntu Bonding](https://help.ubuntu.com/community/UbuntuBonding)

6. 重启网络服务：

    ```bash
    sudo service networking restart
    ```

## FAQ

Q1. 错误信息：

```text
RTNETLINK answers: File exists
Failed to bring up XXX
```

A1. 解决方案：

1. 确认 `/etc/network/interfaces` 中的 gateway 只设置了一次；
2. 执行：

    ```bash
    sudo ip addr flush dev XXX
    ```

Q2. 在 Ubuntu Desktop 下，通过 `/etc/network/interfaces` 配置网络会与 GUI 下的 NetworkManager 服务冲突，我们需要关闭并禁用这个服务：

```bash
sudo systemctl stop NetworkManager.service
sudo systemctl disable NetworkManager.service
```
