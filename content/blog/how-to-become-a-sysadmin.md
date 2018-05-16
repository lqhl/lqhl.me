---
title: 如何成为系统管理员（sysadmin）？
date: 2018-05-31T20:50:11+08:00
description: 关于学习如何成为 Linux 系统管理员的资料
tags: ['linux']
---

## Linux 入门学习

实践课程：

* [Teeny Tiny Linux Server Course](https://github.com/snori74/ebook1/blob/master/course.md)

参考书：

* [鳥哥的 Linux 私房菜 — 基礎學習篇](http://linux.vbird.org/linux_basic/)
* [The Linux Command Line by William E. Shotts, Jr.](http://linuxcommand.org/tlcl.php)
  * [中文版](http://billie66.github.io/TLCL/book/index.html)

寻求帮助：

* 善用 Google 和 Stackoverflow
* RTFM: read the fucking manual!
* Reddit: [r/linuxadmin/](https://www.reddit.com/r/linuxadmin/), [r/linux_mentor](https://www.reddit.com/r/linux_mentor/)

## Linux 的进阶学习

Reddit 上有一个非常好的 list，列出了成为一名 Linux 系统管理员需要学习的内容，可以作为进阶学习的参考：[How did you get your start? : linuxadmin](https://www.reddit.com/r/linuxadmin/comments/2s924h/how_did_you_get_your_start/cnnw1ma/)。

> This is what I tell people to do, who ask me "how do I learn to be a Linux sysadmin?".  
>
> 1) Set up a KVM hypervisor.  
> 2) Inside of that KVM hypervisor, install a Spacewalk server. Use CentOS 6 as the distro for all work below. (For bonus points, set up errata importation on the CentOS channels, so you can properly see security update advisory information.)  
> 3) Create a VM to provide named and dhcpd service to your entire environment. Set up the dhcp daemon to use the Spacewalk server as the pxeboot machine (thus allowing you to use Cobbler to do unattended OS installs). Make sure that every forward zone you create has a reverse zone associated with it. Use something like "internal.virtnet" (but not ".local") as your internal DNS zone.  
> 4) Use that Spacewalk server to automatically (without touching it) install a new pair of OS instances, with which you will then create a Master/Master pair of LDAP servers. Make sure they register with the Spacewalk server. Do not allow anonymous bind, do not use unencrypted LDAP.  
> 5) Reconfigure all 3 servers to use LDAP authentication.  
> 6) Create two new VMs, again unattendedly, which will then be Postgresql VMs. Use pgpool-II to set up master/master replication between them. Export the database from your Spacewalk server and import it into the new pgsql cluster. Reconfigure your Spacewalk instance to run off of that server.  
> 7) Set up a Puppet Master. Plug it into the Spacewalk server for identifying the inventory it will need to work with. (Cheat and use ansible for deployment purposes, again plugging into the Spacewalk server.)  
> 8) Deploy another VM. Install iscsitgt and nfs-kernel-server on it. Export a LUN and an NFS share.  
> 9) Deploy another VM. Install bakula on it, using the postgresql cluster to store its database. Register each machine on it, storing to flatfile. Store the bakula VM's image on the iscsi LUN, and every other machine on the NFS share.  
> 10) Deploy two more VMs. These will have httpd (Apache2) on them. Leave essentially default for now.  
> 11) Deploy two more VMs. These will have tomcat on them. Use JBoss Cache to replicate the session caches between them. Use the httpd servers as the frontends for this. The application you will run is JBoss Wiki.  
> 12) You guessed right, deploy another VM. This will do iptables-based NAT/round-robin loadbalancing between the two httpd servers.  
> 13) Deploy another VM. On this VM, install postfix. Set it up to use a gmail account to allow you to have it send emails, and receive messages only from your internal network.  
> 14) Deploy another VM. On this VM, set up a Nagios server. Have it use snmp to monitor the communication state of every relevant service involved above. This means doing a "is the right port open" check, and a "I got the right kind of response" check and "We still have filesystem space free" check.  
> 15) Deploy another VM. On this VM, set up a syslog daemon to listen to every other server's input. Reconfigure each other server to send their logging output to various files on the syslog server. (For extra credit, set up logstash or kibana or greylog to parse those logs.)  
> 16) Document every last step you did in getting to this point in your brand new Wiki.  
> 17) Now go back and create Puppet Manifests to ensure that every last one of these machines is authenticating to the LDAP servers, registered to the Spacewalk server, and backed up by the bakula server.  
> 18) Now go back, reference your documents, and set up a Puppet Razor profile that hooks into each of these things to allow you to recreate, from scratch, each individual server.  
> 19) Destroy every secondary machine you've created and use the above profile to recreate them, joining them to the clusters as needed.  
> 20) Bonus exercise: create three more VMs. A CentOS 5, 6, and 7 machine. On each of these machines, set them up to allow you to create custom RPMs and import them into the Spacewalk server instance. Ensure your Puppet configurations work for all three and produce like-for-like behaviors.  
>
> Do these things and you will be fully exposed to every aspect of Linux Enterprise systems administration. Do them well and you will have the technical expertise required to seek "Senior" roles. If you go whole-hog crash-course full-time it with no other means of income, I would expect it would take between 3 and 6 months to go from "I think I'm good with computers" to achieving all of these -- assuming you're not afraid of IRC and google (and have neither friends nor family ...).  

## 其它知识

### Cloud 相关知识

学习资料：

* Reddit 上关于学习 Cloud 的路线：[So, you want to learn AWS? AKA, “How do I learn to be a Cloud Engineer?”](https://www.reddit.com/r/sysadmin/comments/8inzn5/so_you_want_to_learn_aws_aka_how_do_i_learn_to_be/)
  * 中文版：[一文掌握AWS，成为云计算工程师](https://mp.weixin.qq.com/s/rD5_EGUv7fth5y6mCxzZ0Q)
* Qwiklabs Labs 提供一系列的实践教程：[Home | Qwiklabs](https://qwiklabs.com/?locale=en)

### 分布式环境下的监控与日志

* 监控机器与服务的状态：Prometheus + Grafana
* 分布式的日志收集与分析：Elasticsearch + Kibana + fluentd

### 实践指南

这节列出一些可供学习的自动部署方案：

* [GitHub - ceph/ceph-ansible: Ansible playbooks for Ceph](https://github.com/ceph/ceph-ansible)
* [TiDB Ansible 部署方案| PingCAP](https://pingcap.com/docs-cn/op-guide/ansible-deployment/)
* [离线 TiDB Ansible 部署方案| PingCAP](https://pingcap.com/docs-cn/op-guide/offline-ansible-deployment/)
* [TiDB中的混沌实践](https://mp.weixin.qq.com/s?__biz=MzIzNjUxMzk2NQ==&mid=2247489285&idx=1&sn=5431d872482793f07404b2428e70dc0d&chksm=e8d7e8c7dfa061d14dc3040e8e63a2b1fb0346b15518178e41ad9511151200e2a7a802958fba&scene=27#wechat_redirect)
