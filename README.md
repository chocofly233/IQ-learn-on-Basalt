尝试复现2021MineRL_BASALT_Competition_Findcave中#2 Team Obsidian采用的方法，即使用IQ_Learn完成矿洞探索，原分享如下：https://www.youtube.com/watch?v=ZOHIFjZB-DM&t=789s。
自己额外使用视频最后五帧作为Cave图片训练了判别器作额外特征，但精度不高，作用不大。Dataloader为其他团队设计的提取视频帧，并随机抽样的工具，不上传到本仓库。
网络结构如下：
![网络流程图](https://github.com/chocofly233/IQ-learn-on-Basalt/blob/main/%E6%B5%81%E7%A8%8B%E5%9B%BE.jpg)
训练设备为我的3060 Laptop，同时16GB RAM不太足以支撑minerl环境下的训练，过程中经常崩溃，训练效果很差。
同时由于该环境特殊，不能使用原作者仓库的IQ-learn算法，所以通过论文写了简易版本的IQ-learn，并在CartPole_v0环境下验证，成功收敛。
![Validation](https://github.com/chocofly233/IQ-learn-on-Basalt/blob/main/iql_validation_detailed.png)
↑这一步非常重要，任何复杂代码都应当在简单环境下得到验证后再进行后续工作。
最终得到训练效果如下：
![Cave](https://github.com/chocofly233/IQ-learn-on-Basalt/blob/main/Cave_1.gif)
![Exploring](https://github.com/chocofly233/IQ-learn-on-Basalt/blob/main/Exploring_1.gif)
