# Round 3 Phase 1 失败案例统计

数据来自 outputs/round3/phase1 中已经完成的 frozen final canonical 结果，协议为 round3_revised，每个 cohort 200 episodes。

统计口径：保留 E0 Stage B、E3 Stage A/B、E5 Stage A/B；排除 Stage A shuffled-goal。逐 episode 只保留至少一个可用条件失败的案例；所有可用条件都成功的 episode 不列入案例表。TwoRoom 的 E3 权重缺失，按 unavailable 处理。

## 总览

| Task | n | Available conditions | Unavailable | Any-failure episodes | All available conditions pass | Any-failure rate | Failure events |
|---|---:|---|---|---:|---:|---:|---:|
| cube | 200 | E0B+E3A+E3B+E5A+E5B | — | 137 | 63 | 68.5% | 315 |
| pusht | 200 | E0B+E3A+E3B+E5A+E5B | — | 50 | 150 | 25.0% | 91 |
| reacher | 200 | E0B+E3A+E3B+E5A+E5B | — | 113 | 87 | 56.5% | 193 |
| tworoom | 200 | E0B+E5A+E5B | E3A+E3B | 38 | 162 | 19.0% | 50 |

合计可用条件-episode 单元：3600；失败事件：649；至少一个条件失败的 episode：338。

## 各条件失败统计

| Task | Condition | Status | n | Success | Failure | Failure rate | Failure with invalid action | Rollout failure |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cube | E0B | ok | 200 | 96 | 104 | 52.0% | 103 | 0 |
| cube | E3A | ok | 200 | 199 | 1 | 0.5% | 1 | 0 |
| cube | E3B | ok | 200 | 92 | 108 | 54.0% | 104 | 0 |
| cube | E5A | ok | 200 | 199 | 1 | 0.5% | 1 | 0 |
| cube | E5B | ok | 200 | 99 | 101 | 50.5% | 100 | 0 |
| pusht | E0B | ok | 200 | 186 | 14 | 7.0% | 9 | 0 |
| pusht | E3A | ok | 200 | 188 | 12 | 6.0% | 0 | 0 |
| pusht | E3B | ok | 200 | 169 | 31 | 15.5% | 9 | 0 |
| pusht | E5A | ok | 200 | 187 | 13 | 6.5% | 0 | 0 |
| pusht | E5B | ok | 200 | 179 | 21 | 10.5% | 7 | 0 |
| reacher | E0B | ok | 200 | 169 | 31 | 15.5% | 24 | 0 |
| reacher | E3A | ok | 200 | 141 | 59 | 29.5% | 1 | 0 |
| reacher | E3B | ok | 200 | 169 | 31 | 15.5% | 23 | 0 |
| reacher | E5A | ok | 200 | 159 | 41 | 20.5% | 5 | 0 |
| reacher | E5B | ok | 200 | 169 | 31 | 15.5% | 22 | 0 |
| tworoom | E0B | ok | 200 | 169 | 31 | 15.5% | 31 | 0 |
| tworoom | E3A | unavailable | — | — | — | — | — | — |
| tworoom | E3B | unavailable | — | — | — | — | — | — |
| tworoom | E5A | ok | 200 | 190 | 10 | 5.0% | 10 | 0 |
| tworoom | E5B | ok | 200 | 191 | 9 | 4.5% | 9 | 0 |

## 失败条件组合

组合只统计至少一个条件失败的 episode；例如 E0B+E3B 表示该 episode 在两个条件下失败。

| Task | Failed conditions | Episodes | Share of task failure cases |
|---|---|---:|---:|
| cube | E0B+E3B+E5B | 75 | 54.7% |
| cube | E0B | 19 | 13.9% |
| cube | E3B+E5B | 15 | 10.9% |
| cube | E3B | 14 | 10.2% |
| cube | E0B+E5B | 6 | 4.4% |
| cube | E5B | 4 | 2.9% |
| cube | E0B+E3B | 3 | 2.2% |
| cube | E0B+E3A+E3B+E5A+E5B | 1 | 0.7% |
| pusht | E3B | 12 | 24.0% |
| pusht | E0B+E3B+E5B | 6 | 12.0% |
| pusht | E5B | 5 | 10.0% |
| pusht | E0B | 4 | 8.0% |
| pusht | E3B+E5B | 4 | 8.0% |
| pusht | E3A | 3 | 6.0% |
| pusht | E5A | 3 | 6.0% |
| pusht | E0B+E3B | 2 | 4.0% |
| pusht | E3A+E3B+E5A+E5B | 2 | 4.0% |
| pusht | E3A+E5A | 2 | 4.0% |
| pusht | E0B+E3A+E3B+E5A | 1 | 2.0% |
| pusht | E0B+E3A+E3B+E5A+E5B | 1 | 2.0% |
| pusht | E3A+E3B | 1 | 2.0% |
| pusht | E3A+E3B+E5A | 1 | 2.0% |
| pusht | E3A+E5A+E5B | 1 | 2.0% |
| pusht | E3B+E5A+E5B | 1 | 2.0% |
| pusht | E5A+E5B | 1 | 2.0% |
| reacher | E3A | 17 | 15.0% |
| reacher | E5B | 14 | 12.4% |
| reacher | E3A+E5A | 12 | 10.6% |
| reacher | E5A | 12 | 10.6% |
| reacher | E3B | 11 | 9.7% |
| reacher | E0B+E3A+E5A | 6 | 5.3% |
| reacher | E0B | 5 | 4.4% |
| reacher | E0B+E3A | 4 | 3.5% |
| reacher | E0B+E3B | 4 | 3.5% |
| reacher | E3A+E3B | 4 | 3.5% |
| reacher | E3A+E5B | 4 | 3.5% |
| reacher | E0B+E3A+E3B | 2 | 1.8% |
| reacher | E0B+E3A+E5A+E5B | 2 | 1.8% |
| reacher | E0B+E5A | 2 | 1.8% |
| reacher | E3A+E3B+E5A | 2 | 1.8% |
| reacher | E3A+E5A+E5B | 2 | 1.8% |
| reacher | E3B+E5B | 2 | 1.8% |
| reacher | E0B+E3A+E3B+E5B | 1 | 0.9% |
| reacher | E0B+E3A+E5B | 1 | 0.9% |
| reacher | E0B+E3B+E5A | 1 | 0.9% |
| reacher | E0B+E3B+E5A+E5B | 1 | 0.9% |
| reacher | E0B+E3B+E5B | 1 | 0.9% |
| reacher | E0B+E5B | 1 | 0.9% |
| reacher | E3A+E3B+E5A+E5B | 1 | 0.9% |
| reacher | E3A+E3B+E5B | 1 | 0.9% |
| tworoom | E0B | 20 | 52.6% |
| tworoom | E0B+E5B | 7 | 18.4% |
| tworoom | E5A | 6 | 15.8% |
| tworoom | E0B+E5A | 3 | 7.9% |
| tworoom | E0B+E5A+E5B | 1 | 2.6% |
| tworoom | E5B | 1 | 2.6% |

## 逐 episode 失败案例

完整明细见 failure_cases.csv。表中 failure/success 是对应条件的 frozen final 结果；*_video 仅在独立视频重跑已成功发布且文件存在时填写。Reacher 视频重跑已暂停，因此 Reacher 的 failure analysis 仍有 canonical 结果，但视频列暂为空。

### cube（137 个至少失败一次的 episode）

| Slot | Dataset episode | Row | Start step | Goal step | Failed conditions | E0B | E3A | E3B | E5A | E5B |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| 0 | 16 | 3228 | 12 | 37 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 2 | 108 | 21818 | 110 | 135 | E3B+E5B | success | success | failure | success | failure |
| 3 | 146 | 29496 | 150 | 175 | E3B | success | success | failure | success | success |
| 5 | 190 | 38226 | 36 | 61 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 6 | 290 | 58340 | 50 | 75 | E0B | failure | success | success | success | success |
| 7 | 348 | 69981 | 33 | 58 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 8 | 375 | 75392 | 17 | 42 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 9 | 434 | 87360 | 126 | 151 | E0B | failure | success | success | success | success |
| 12 | 585 | 117596 | 11 | 36 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 14 | 832 | 167335 | 103 | 128 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 18 | 976 | 196191 | 15 | 40 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 19 | 1136 | 228379 | 43 | 68 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 20 | 1218 | 244861 | 43 | 68 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 21 | 1261 | 253472 | 11 | 36 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 23 | 1401 | 281626 | 25 | 50 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 24 | 1414 | 284352 | 138 | 163 | E0B+E3B | failure | success | failure | success | success |
| 25 | 1436 | 288652 | 16 | 41 | E3B+E5B | success | success | failure | success | failure |
| 26 | 1441 | 289761 | 120 | 145 | E0B | failure | success | success | success | success |
| 28 | 1884 | 378782 | 98 | 123 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 30 | 1919 | 385749 | 30 | 55 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 32 | 1991 | 400321 | 130 | 155 | E0B | failure | success | success | success | success |
| 33 | 2008 | 403757 | 149 | 174 | E3B | success | success | failure | success | success |
| 38 | 2425 | 487473 | 48 | 73 | E3B+E5B | success | success | failure | success | failure |
| 39 | 2465 | 495483 | 18 | 43 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 40 | 2481 | 498727 | 46 | 71 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 41 | 2482 | 498988 | 106 | 131 | E0B | failure | success | success | success | success |
| 44 | 2633 | 529245 | 12 | 37 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 45 | 2676 | 537985 | 109 | 134 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 50 | 2954 | 593796 | 42 | 67 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 52 | 2984 | 599800 | 16 | 41 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 53 | 3004 | 603818 | 14 | 39 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 55 | 3143 | 631889 | 146 | 171 | E3B+E5B | success | success | failure | success | failure |
| 56 | 3182 | 639627 | 45 | 70 | E0B | failure | success | success | success | success |
| 57 | 3222 | 647747 | 125 | 150 | E0B | failure | success | success | success | success |
| 58 | 3358 | 675106 | 148 | 173 | E3B+E5B | success | success | failure | success | failure |
| 59 | 3402 | 683913 | 111 | 136 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 61 | 3471 | 697770 | 99 | 124 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 62 | 3548 | 713253 | 105 | 130 | E3B+E5B | success | success | failure | success | failure |
| 63 | 3579 | 719483 | 104 | 129 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 65 | 3626 | 728963 | 137 | 162 | E0B | failure | success | success | success | success |
| 68 | 3695 | 742708 | 13 | 38 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 69 | 3727 | 749146 | 19 | 44 | E3B | success | success | failure | success | success |
| 70 | 3809 | 765666 | 57 | 82 | E3B | success | success | failure | success | success |
| 71 | 3846 | 773075 | 29 | 54 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 72 | 3877 | 779318 | 41 | 66 | E3B | success | success | failure | success | success |
| 73 | 3898 | 783598 | 100 | 125 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 75 | 3963 | 796619 | 56 | 81 | E0B | failure | success | success | success | success |
| 76 | 4015 | 807160 | 145 | 170 | E3B | success | success | failure | success | success |
| 77 | 4058 | 815676 | 18 | 43 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 79 | 4202 | 844700 | 98 | 123 | E3B+E5B | success | success | failure | success | failure |
| 80 | 4213 | 846832 | 19 | 44 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 81 | 4218 | 847841 | 23 | 48 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 83 | 4389 | 882207 | 18 | 43 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 85 | 4443 | 893067 | 24 | 49 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 87 | 4497 | 904012 | 115 | 140 | E3B | success | success | failure | success | success |
| 88 | 4657 | 936107 | 50 | 75 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 89 | 4692 | 943122 | 30 | 55 | E0B | failure | success | success | success | success |
| 92 | 4850 | 974957 | 107 | 132 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 93 | 4858 | 976559 | 101 | 126 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 94 | 4945 | 993954 | 9 | 34 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 95 | 5011 | 1007356 | 145 | 170 | E3B | success | success | failure | success | success |
| 96 | 5029 | 1010878 | 49 | 74 | E3B+E5B | success | success | failure | success | failure |
| 97 | 5071 | 1019279 | 8 | 33 | E0B+E3A+E3B+E5A+E5B | failure | failure | failure | failure | failure |
| 98 | 5121 | 1029432 | 111 | 136 | E0B | failure | success | success | success | success |
| 99 | 5241 | 1053483 | 42 | 67 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 100 | 5242 | 1053680 | 38 | 63 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 101 | 5293 | 1063949 | 56 | 81 | E5B | success | success | success | success | failure |
| 102 | 5324 | 1070134 | 10 | 35 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 103 | 5405 | 1086539 | 134 | 159 | E3B+E5B | success | success | failure | success | failure |
| 104 | 5440 | 1093493 | 53 | 78 | E0B | failure | success | success | success | success |
| 105 | 5441 | 1093656 | 15 | 40 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 106 | 5480 | 1101522 | 42 | 67 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 107 | 5590 | 1123707 | 117 | 142 | E5B | success | success | success | success | failure |
| 108 | 5699 | 1145622 | 123 | 148 | E0B | failure | success | success | success | success |
| 109 | 5715 | 1148759 | 44 | 69 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 110 | 5736 | 1153073 | 137 | 162 | E0B+E5B | failure | success | success | success | failure |
| 111 | 5753 | 1156375 | 22 | 47 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 112 | 5762 | 1158185 | 23 | 48 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 114 | 5824 | 1170668 | 44 | 69 | E0B+E5B | failure | success | success | success | failure |
| 115 | 5850 | 1175869 | 19 | 44 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 116 | 5856 | 1177157 | 101 | 126 | E0B+E3B | failure | success | failure | success | success |
| 118 | 5961 | 1198260 | 99 | 124 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 121 | 6204 | 1247111 | 107 | 132 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 122 | 6208 | 1247836 | 28 | 53 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 123 | 6270 | 1260329 | 59 | 84 | E0B+E3B | failure | success | failure | success | success |
| 124 | 6296 | 1265522 | 26 | 51 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 126 | 6445 | 1295471 | 26 | 51 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 127 | 6450 | 1296583 | 133 | 158 | E3B+E5B | success | success | failure | success | failure |
| 130 | 6632 | 1333161 | 129 | 154 | E3B | success | success | failure | success | success |
| 131 | 6659 | 1338491 | 32 | 57 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 133 | 6740 | 1354882 | 142 | 167 | E3B | success | success | failure | success | success |
| 135 | 6787 | 1364201 | 14 | 39 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 136 | 6805 | 1367907 | 102 | 127 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 139 | 6991 | 1405301 | 110 | 135 | E3B+E5B | success | success | failure | success | failure |
| 142 | 7155 | 1438196 | 41 | 66 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 143 | 7156 | 1438382 | 26 | 51 | E3B | success | success | failure | success | success |
| 146 | 7331 | 1473628 | 97 | 122 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 147 | 7369 | 1481273 | 104 | 129 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 148 | 7397 | 1486932 | 135 | 160 | E3B+E5B | success | success | failure | success | failure |
| 149 | 7469 | 1501282 | 13 | 38 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 150 | 7576 | 1522915 | 139 | 164 | E3B | success | success | failure | success | success |
| 152 | 7731 | 1553969 | 38 | 63 | E0B | failure | success | success | success | success |
| 153 | 7735 | 1554849 | 114 | 139 | E3B | success | success | failure | success | success |
| 154 | 7738 | 1555479 | 141 | 166 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 156 | 7859 | 1579688 | 29 | 54 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 157 | 7889 | 1585718 | 29 | 54 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 158 | 7950 | 1598071 | 121 | 146 | E0B | failure | success | success | success | success |
| 159 | 8046 | 1617348 | 102 | 127 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 160 | 8080 | 1624089 | 9 | 34 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 161 | 8105 | 1629140 | 35 | 60 | E0B+E5B | failure | success | success | success | failure |
| 163 | 8180 | 1644216 | 36 | 61 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 166 | 8334 | 1675143 | 9 | 34 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 167 | 8340 | 1676380 | 40 | 65 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 168 | 8342 | 1676765 | 23 | 48 | E0B+E5B | failure | success | success | success | failure |
| 173 | 8791 | 1767110 | 119 | 144 | E0B | failure | success | success | success | success |
| 174 | 8894 | 1787712 | 18 | 43 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 175 | 8935 | 1795976 | 41 | 66 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 176 | 9113 | 1831815 | 102 | 127 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 177 | 9116 | 1832342 | 26 | 51 | E3B | success | success | failure | success | success |
| 179 | 9119 | 1833041 | 122 | 147 | E5B | success | success | success | success | failure |
| 180 | 9139 | 1836988 | 49 | 74 | E0B+E5B | failure | success | success | success | failure |
| 181 | 9214 | 1852055 | 41 | 66 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 182 | 9342 | 1877768 | 26 | 51 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 184 | 9513 | 1912134 | 21 | 46 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 185 | 9519 | 1913339 | 20 | 45 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 186 | 9529 | 1915425 | 96 | 121 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 187 | 9593 | 1928303 | 110 | 135 | E0B+E5B | failure | success | success | success | failure |
| 188 | 9609 | 1931459 | 50 | 75 | E3B+E5B | success | success | failure | success | failure |
| 190 | 9715 | 1952762 | 47 | 72 | E3B+E5B | success | success | failure | success | failure |
| 191 | 9735 | 1956843 | 108 | 133 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 193 | 9780 | 1965918 | 138 | 163 | E0B | failure | success | success | success | success |
| 194 | 9830 | 1975870 | 40 | 65 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 195 | 9845 | 1978946 | 101 | 126 | E0B | failure | success | success | success | success |
| 196 | 9896 | 1989118 | 22 | 47 | E3B+E5B | success | success | failure | success | failure |
| 197 | 9913 | 1992523 | 10 | 35 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 198 | 9929 | 1995837 | 108 | 133 | E0B | failure | success | success | success | success |
| 199 | 9950 | 1999995 | 45 | 70 | E5B | success | success | success | success | failure |

### pusht（50 个至少失败一次的 episode）

| Slot | Dataset episode | Row | Start step | Goal step | Failed conditions | E0B | E3A | E3B | E5A | E5B |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| 1 | 179 | 21250 | 23 | 48 | E3B | success | success | failure | success | success |
| 3 | 250 | 28316 | 44 | 69 | E5B | success | success | success | success | failure |
| 6 | 529 | 56315 | 64 | 89 | E5A | success | success | success | failure | success |
| 9 | 578 | 62430 | 5 | 30 | E3B | success | success | failure | success | success |
| 10 | 619 | 68026 | 19 | 44 | E3A | success | failure | success | success | success |
| 12 | 956 | 118978 | 23 | 48 | E3B+E5B | success | success | failure | success | failure |
| 13 | 1413 | 175186 | 9 | 34 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 14 | 1576 | 193043 | 58 | 83 | E0B+E3A+E3B+E5A | failure | failure | failure | failure | success |
| 15 | 2004 | 240406 | 6 | 31 | E3B | success | success | failure | success | success |
| 18 | 2236 | 266422 | 44 | 69 | E3B+E5B | success | success | failure | success | failure |
| 21 | 2462 | 290516 | 14 | 39 | E5B | success | success | success | success | failure |
| 28 | 3431 | 420321 | 67 | 92 | E3B | success | success | failure | success | success |
| 30 | 3638 | 444192 | 5 | 30 | E3B | success | success | failure | success | success |
| 31 | 3684 | 451015 | 112 | 137 | E5A | success | success | success | failure | success |
| 32 | 4085 | 510376 | 74 | 99 | E0B | failure | success | success | success | success |
| 34 | 4332 | 535057 | 53 | 78 | E3A | success | failure | success | success | success |
| 35 | 4344 | 536043 | 38 | 63 | E5B | success | success | success | success | failure |
| 38 | 4834 | 598249 | 78 | 103 | E3A+E5A | success | failure | success | failure | success |
| 40 | 4971 | 618649 | 109 | 134 | E3B+E5B | success | success | failure | success | failure |
| 49 | 5746 | 717937 | 82 | 107 | E0B | failure | success | success | success | success |
| 50 | 5896 | 735208 | 61 | 86 | E3B | success | success | failure | success | success |
| 67 | 7350 | 910690 | 2 | 27 | E0B+E3B | failure | success | failure | success | success |
| 68 | 7371 | 913504 | 23 | 48 | E3A+E3B+E5A+E5B | success | failure | failure | failure | failure |
| 76 | 7966 | 979358 | 13 | 38 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 79 | 8115 | 998877 | 64 | 89 | E3B+E5A+E5B | success | success | failure | failure | failure |
| 91 | 9364 | 1150667 | 0 | 25 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 92 | 9393 | 1153029 | 13 | 38 | E0B | failure | success | success | success | success |
| 94 | 9456 | 1158674 | 51 | 76 | E3A | success | failure | success | success | success |
| 97 | 9802 | 1198227 | 35 | 60 | E3B | success | success | failure | success | success |
| 98 | 9997 | 1224410 | 67 | 92 | E3A+E3B | success | failure | failure | success | success |
| 106 | 10627 | 1312153 | 95 | 120 | E0B+E3B | failure | success | failure | success | success |
| 108 | 10800 | 1333180 | 60 | 85 | E5A | success | success | success | failure | success |
| 110 | 10994 | 1361092 | 39 | 64 | E5A+E5B | success | success | success | failure | failure |
| 120 | 11701 | 1466575 | 51 | 76 | E0B+E3A+E3B+E5A+E5B | failure | failure | failure | failure | failure |
| 139 | 13307 | 1674610 | 118 | 143 | E3B | success | success | failure | success | success |
| 140 | 13322 | 1676811 | 24 | 49 | E3A+E3B+E5A+E5B | success | failure | failure | failure | failure |
| 150 | 14301 | 1802777 | 63 | 88 | E0B | failure | success | success | success | success |
| 152 | 14371 | 1809625 | 65 | 90 | E5B | success | success | success | success | failure |
| 159 | 14995 | 1883140 | 40 | 65 | E3B+E5B | success | success | failure | success | failure |
| 160 | 15107 | 1895941 | 15 | 40 | E5B | success | success | success | success | failure |
| 162 | 15357 | 1927769 | 36 | 61 | E3B | success | success | failure | success | success |
| 164 | 15445 | 1933902 | 9 | 34 | E3A+E5A+E5B | success | failure | success | failure | failure |
| 171 | 16077 | 2016277 | 3 | 28 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 172 | 16100 | 2020807 | 2 | 27 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 175 | 16731 | 2108183 | 39 | 64 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 184 | 17134 | 2146733 | 46 | 71 | E3B | success | success | failure | success | success |
| 187 | 17463 | 2195038 | 15 | 40 | E3A+E5A | success | failure | success | failure | success |
| 193 | 18069 | 2276701 | 41 | 66 | E3B | success | success | failure | success | success |
| 198 | 18453 | 2311895 | 57 | 82 | E3A+E3B+E5A | success | failure | failure | failure | success |
| 199 | 18510 | 2317556 | 3 | 28 | E3B | success | success | failure | success | success |

### reacher（113 个至少失败一次的 episode）

| Slot | Dataset episode | Row | Start step | Goal step | Failed conditions | E0B | E3A | E3B | E5A | E5B |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| 6 | 286 | 57644 | 158 | 183 | E5A | success | success | success | failure | success |
| 8 | 348 | 69981 | 33 | 58 | E5B | success | success | success | success | failure |
| 10 | 424 | 85399 | 175 | 200 | E0B+E3A+E5A | failure | failure | success | failure | success |
| 12 | 585 | 117596 | 11 | 36 | E3A+E5B | success | failure | success | success | failure |
| 14 | 715 | 143879 | 164 | 189 | E0B+E5B | failure | success | success | success | failure |
| 15 | 832 | 167335 | 103 | 128 | E5B | success | success | success | success | failure |
| 16 | 896 | 180150 | 54 | 79 | E3A+E5B | success | failure | success | success | failure |
| 17 | 898 | 180549 | 51 | 76 | E3A+E5A | success | failure | success | failure | success |
| 18 | 963 | 193733 | 170 | 195 | E5A | success | success | success | failure | success |
| 20 | 1083 | 217841 | 158 | 183 | E3A | success | failure | success | success | success |
| 22 | 1218 | 244861 | 43 | 68 | E3B+E5B | success | success | failure | success | failure |
| 26 | 1626 | 326829 | 3 | 28 | E0B | failure | success | success | success | success |
| 28 | 1736 | 349003 | 67 | 92 | E0B | failure | success | success | success | success |
| 30 | 1919 | 385749 | 30 | 55 | E3B | success | success | failure | success | success |
| 31 | 2008 | 403757 | 149 | 174 | E5A | success | success | success | failure | success |
| 32 | 2167 | 435721 | 154 | 179 | E0B+E3B | failure | success | failure | success | success |
| 33 | 2177 | 437661 | 84 | 109 | E0B+E5A | failure | success | success | failure | success |
| 34 | 2210 | 444338 | 128 | 153 | E5A | success | success | success | failure | success |
| 35 | 2218 | 445956 | 138 | 163 | E3A | success | failure | success | success | success |
| 38 | 2324 | 467236 | 112 | 137 | E3A+E3B+E5A+E5B | success | failure | failure | failure | failure |
| 40 | 2425 | 487473 | 48 | 73 | E3A | success | failure | success | success | success |
| 42 | 2482 | 498988 | 106 | 131 | E3A | success | failure | success | success | success |
| 43 | 2547 | 512097 | 150 | 175 | E0B+E3A+E5A | failure | failure | success | failure | success |
| 44 | 2633 | 529245 | 12 | 37 | E0B+E3A+E5A | failure | failure | success | failure | success |
| 46 | 2654 | 533532 | 78 | 103 | E5A | success | success | success | failure | success |
| 49 | 2899 | 582703 | 4 | 29 | E3A+E3B | success | failure | failure | success | success |
| 50 | 2905 | 583963 | 58 | 83 | E3A | success | failure | success | success | success |
| 55 | 3160 | 635164 | 4 | 29 | E5B | success | success | success | success | failure |
| 56 | 3192 | 641633 | 41 | 66 | E3B | success | success | failure | success | success |
| 60 | 3346 | 672689 | 143 | 168 | E5B | success | success | success | success | failure |
| 61 | 3358 | 675106 | 148 | 173 | E5B | success | success | success | success | failure |
| 63 | 3448 | 693091 | 43 | 68 | E0B | failure | success | success | success | success |
| 64 | 3471 | 697770 | 99 | 124 | E3A+E5A | success | failure | success | failure | success |
| 66 | 3548 | 713253 | 105 | 130 | E0B+E3A | failure | failure | success | success | success |
| 67 | 3570 | 717739 | 169 | 194 | E3A | success | failure | success | success | success |
| 69 | 3586 | 720929 | 143 | 168 | E0B+E3B+E5A | failure | success | failure | failure | success |
| 71 | 3667 | 737104 | 37 | 62 | E0B+E3A+E3B | failure | failure | failure | success | success |
| 74 | 3846 | 773075 | 29 | 54 | E5B | success | success | success | success | failure |
| 75 | 3877 | 779318 | 41 | 66 | E3A+E3B+E5B | success | failure | failure | success | failure |
| 76 | 3907 | 785313 | 6 | 31 | E3A+E5A | success | failure | success | failure | success |
| 77 | 4191 | 842519 | 128 | 153 | E3B | success | success | failure | success | success |
| 78 | 4202 | 844700 | 98 | 123 | E3B | success | success | failure | success | success |
| 80 | 4453 | 895197 | 144 | 169 | E3A+E5A | success | failure | success | failure | success |
| 81 | 4497 | 904012 | 115 | 140 | E0B+E3B | failure | success | failure | success | success |
| 82 | 4552 | 915110 | 158 | 183 | E3A+E5A | success | failure | success | failure | success |
| 83 | 4567 | 918039 | 72 | 97 | E3A+E5A+E5B | success | failure | success | failure | failure |
| 85 | 4657 | 936107 | 50 | 75 | E3A | success | failure | success | success | success |
| 87 | 4828 | 970599 | 171 | 196 | E5A | success | success | success | failure | success |
| 88 | 4945 | 993954 | 9 | 34 | E5B | success | success | success | success | failure |
| 92 | 5294 | 1064267 | 173 | 198 | E3B | success | success | failure | success | success |
| 93 | 5354 | 1076326 | 172 | 197 | E3A+E5A | success | failure | success | failure | success |
| 94 | 5398 | 1085029 | 31 | 56 | E0B | failure | success | success | success | success |
| 98 | 5480 | 1101522 | 42 | 67 | E3A+E3B+E5A | success | failure | failure | failure | success |
| 99 | 5746 | 1155108 | 162 | 187 | E3B | success | success | failure | success | success |
| 103 | 5766 | 1159016 | 50 | 75 | E0B+E3A+E5A+E5B | failure | failure | success | failure | failure |
| 104 | 5798 | 1165480 | 82 | 107 | E3A+E3B | success | failure | failure | success | success |
| 107 | 5856 | 1177157 | 101 | 126 | E5B | success | success | success | success | failure |
| 109 | 5924 | 1190898 | 174 | 199 | E5A | success | success | success | failure | success |
| 110 | 6041 | 1214317 | 76 | 101 | E3A+E5A+E5B | success | failure | success | failure | failure |
| 112 | 6175 | 1241346 | 171 | 196 | E3A | success | failure | success | success | success |
| 113 | 6204 | 1247111 | 107 | 132 | E0B+E3A+E5A+E5B | failure | failure | success | failure | failure |
| 114 | 6208 | 1247836 | 28 | 53 | E3A+E5A | success | failure | success | failure | success |
| 118 | 6335 | 1273452 | 117 | 142 | E0B+E3A+E5A | failure | failure | success | failure | success |
| 119 | 6339 | 1274307 | 168 | 193 | E3A+E3B+E5A | success | failure | failure | failure | success |
| 120 | 6359 | 1278233 | 74 | 99 | E3A | success | failure | success | success | success |
| 121 | 6450 | 1296583 | 133 | 158 | E3A+E5A | success | failure | success | failure | success |
| 124 | 6537 | 1313978 | 41 | 66 | E3A | success | failure | success | success | success |
| 125 | 6601 | 1326954 | 153 | 178 | E3A+E5A | success | failure | success | failure | success |
| 127 | 6660 | 1338688 | 28 | 53 | E3A | success | failure | success | success | success |
| 132 | 6873 | 1381568 | 95 | 120 | E0B+E3A | failure | failure | success | success | success |
| 133 | 6891 | 1385166 | 75 | 100 | E0B+E3B+E5B | failure | success | failure | success | failure |
| 134 | 6981 | 1403297 | 116 | 141 | E5B | success | success | success | success | failure |
| 135 | 6991 | 1405301 | 110 | 135 | E3A | success | failure | success | success | success |
| 136 | 7012 | 1409481 | 69 | 94 | E3A+E3B | success | failure | failure | success | success |
| 138 | 7155 | 1438196 | 41 | 66 | E5B | success | success | success | success | failure |
| 140 | 7233 | 1453910 | 77 | 102 | E5B | success | success | success | success | failure |
| 141 | 7310 | 1469396 | 86 | 111 | E0B+E3B+E5A+E5B | failure | success | failure | failure | failure |
| 142 | 7325 | 1472482 | 157 | 182 | E3A+E5B | success | failure | success | success | failure |
| 143 | 7355 | 1478521 | 166 | 191 | E5A | success | success | success | failure | success |
| 144 | 7394 | 1486272 | 78 | 103 | E5A | success | success | success | failure | success |
| 145 | 7397 | 1486932 | 135 | 160 | E5A | success | success | success | failure | success |
| 146 | 7469 | 1501282 | 13 | 38 | E5B | success | success | success | success | failure |
| 147 | 7515 | 1510537 | 22 | 47 | E0B+E3A+E5B | failure | failure | success | success | failure |
| 148 | 7526 | 1512833 | 107 | 132 | E0B+E3A+E3B+E5B | failure | failure | failure | success | failure |
| 151 | 7637 | 1535211 | 174 | 199 | E5B | success | success | success | success | failure |
| 152 | 7670 | 1541837 | 167 | 192 | E0B+E3A+E5A | failure | failure | success | failure | success |
| 153 | 7731 | 1553969 | 38 | 63 | E0B+E3A | failure | failure | success | success | success |
| 154 | 7738 | 1555479 | 141 | 166 | E0B+E3B | failure | success | failure | success | success |
| 155 | 7767 | 1561287 | 120 | 145 | E0B+E3A+E3B | failure | failure | failure | success | success |
| 156 | 7842 | 1576408 | 166 | 191 | E3B | success | success | failure | success | success |
| 159 | 8046 | 1617348 | 102 | 127 | E0B+E3B | failure | success | failure | success | success |
| 160 | 8054 | 1619020 | 166 | 191 | E3A+E3B | success | failure | failure | success | success |
| 161 | 8065 | 1621240 | 175 | 200 | E3B | success | success | failure | success | success |
| 163 | 8108 | 1629714 | 6 | 31 | E0B | failure | success | success | success | success |
| 164 | 8223 | 1652944 | 121 | 146 | E3B+E5B | success | success | failure | success | failure |
| 166 | 8342 | 1676765 | 23 | 48 | E5A | success | success | success | failure | success |
| 168 | 8609 | 1730550 | 141 | 166 | E3A | success | failure | success | success | success |
| 169 | 8710 | 1750825 | 115 | 140 | E0B+E5A | failure | success | success | failure | success |
| 174 | 8863 | 1781547 | 84 | 109 | E3A+E5A | success | failure | success | failure | success |
| 176 | 9078 | 1824850 | 172 | 197 | E3B | success | success | failure | success | success |
| 177 | 9116 | 1832342 | 26 | 51 | E3A | success | failure | success | success | success |
| 178 | 9117 | 1832645 | 128 | 153 | E3A | success | failure | success | success | success |
| 181 | 9143 | 1837893 | 150 | 175 | E0B+E3A | failure | failure | success | success | success |
| 182 | 9292 | 1867850 | 158 | 183 | E5B | success | success | success | success | failure |
| 184 | 9365 | 1882535 | 170 | 195 | E0B+E3A+E5A | failure | failure | success | failure | success |
| 189 | 9587 | 1926998 | 11 | 36 | E3A | success | failure | success | success | success |
| 190 | 9593 | 1928303 | 110 | 135 | E3B | success | success | failure | success | success |
| 191 | 9649 | 1939610 | 161 | 186 | E3A | success | failure | success | success | success |
| 193 | 9672 | 1944089 | 17 | 42 | E3A+E5B | success | failure | success | success | failure |
| 194 | 9735 | 1956843 | 108 | 133 | E3A+E5A | success | failure | success | failure | success |
| 196 | 9830 | 1975870 | 40 | 65 | E5A | success | success | success | failure | success |
| 198 | 9929 | 1995837 | 108 | 133 | E3B | success | success | failure | success | success |
| 199 | 9950 | 1999995 | 45 | 70 | E3A+E5A | success | failure | success | failure | success |

### tworoom（38 个至少失败一次的 episode）

| Slot | Dataset episode | Row | Start step | Goal step | Failed conditions | E0B | E3A | E3B | E5A | E5B |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| 3 | 143 | 13020 | 9 | 34 | E0B+E5B | failure | unavailable | unavailable | success | failure |
| 5 | 290 | 26585 | 22 | 47 | E0B | failure | unavailable | unavailable | success | success |
| 14 | 898 | 81821 | 22 | 47 | E5A | success | unavailable | unavailable | failure | success |
| 15 | 963 | 87940 | 67 | 92 | E0B | failure | unavailable | unavailable | success | success |
| 18 | 1136 | 103684 | 18 | 43 | E0B+E5B | failure | unavailable | unavailable | success | failure |
| 19 | 1218 | 111553 | 18 | 43 | E0B | failure | unavailable | unavailable | success | success |
| 29 | 1919 | 176187 | 13 | 38 | E0B | failure | unavailable | unavailable | success | success |
| 31 | 2210 | 202968 | 31 | 56 | E5A | success | unavailable | unavailable | failure | success |
| 43 | 2647 | 243196 | 25 | 50 | E0B | failure | unavailable | unavailable | success | success |
| 47 | 2857 | 262833 | 55 | 80 | E0B+E5B | failure | unavailable | unavailable | success | failure |
| 48 | 2899 | 266600 | 1 | 26 | E5B | success | unavailable | unavailable | success | failure |
| 49 | 2905 | 267110 | 25 | 50 | E0B | failure | unavailable | unavailable | success | success |
| 58 | 3402 | 312685 | 39 | 64 | E0B+E5B | failure | unavailable | unavailable | success | failure |
| 79 | 4497 | 413389 | 50 | 75 | E0B | failure | unavailable | unavailable | success | success |
| 81 | 4567 | 419985 | 23 | 48 | E5A | success | unavailable | unavailable | failure | success |
| 90 | 5121 | 470963 | 35 | 60 | E0B+E5B | failure | unavailable | unavailable | success | failure |
| 91 | 5294 | 486935 | 38 | 63 | E0B | failure | unavailable | unavailable | success | success |
| 92 | 5299 | 487401 | 36 | 61 | E0B | failure | unavailable | unavailable | success | success |
| 93 | 5354 | 492598 | 69 | 94 | E0B | failure | unavailable | unavailable | success | success |
| 96 | 5453 | 501839 | 17 | 42 | E0B+E5A | failure | unavailable | unavailable | failure | success |
| 99 | 5715 | 525614 | 19 | 44 | E0B | failure | unavailable | unavailable | success | success |
| 102 | 5755 | 529365 | 57 | 82 | E0B | failure | unavailable | unavailable | success | success |
| 106 | 5798 | 533334 | 29 | 54 | E0B | failure | unavailable | unavailable | success | success |
| 112 | 5961 | 548603 | 33 | 58 | E0B | failure | unavailable | unavailable | success | success |
| 117 | 6270 | 577107 | 25 | 50 | E0B+E5B | failure | unavailable | unavailable | success | failure |
| 119 | 6302 | 580150 | 60 | 85 | E0B+E5A | failure | unavailable | unavailable | failure | success |
| 121 | 6335 | 583107 | 50 | 75 | E5A | success | unavailable | unavailable | failure | success |
| 127 | 6522 | 600109 | 64 | 89 | E0B | failure | unavailable | unavailable | success | success |
| 141 | 7155 | 657887 | 11 | 36 | E0B | failure | unavailable | unavailable | success | success |
| 154 | 7592 | 697993 | 32 | 57 | E0B+E5A+E5B | failure | unavailable | unavailable | failure | failure |
| 160 | 7764 | 714038 | 30 | 55 | E5A | success | unavailable | unavailable | failure | success |
| 163 | 7988 | 734809 | 14 | 39 | E5A | success | unavailable | unavailable | failure | success |
| 165 | 8046 | 739979 | 37 | 62 | E0B+E5A | failure | unavailable | unavailable | failure | success |
| 179 | 9078 | 835032 | 63 | 88 | E0B | failure | unavailable | unavailable | success | success |
| 191 | 9649 | 888130 | 60 | 85 | E0B | failure | unavailable | unavailable | success | success |
| 193 | 9715 | 894297 | 20 | 45 | E0B | failure | unavailable | unavailable | success | success |
| 196 | 9780 | 900469 | 57 | 82 | E0B+E5B | failure | unavailable | unavailable | success | failure |
| 199 | 9950 | 916273 | 19 | 44 | E0B | failure | unavailable | unavailable | success | success |

## 文件

- summary.csv：按 task 的总体统计。
- condition_stats.csv：按 task/condition 的成功失败统计。
- failure_patterns.csv：失败条件组合统计。
- failure_cases.csv：逐 episode 失败案例、诊断字段和可用视频路径。
