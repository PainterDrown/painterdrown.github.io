[painterdrown Blog](https://painterdrown.github.io) - [painterdrown SAAD](https://painterdrown.github.io/saad)

# SAAD Assignment 9 ECB

> ⏰ 2018-06-29 09:33:27<br/>
> 👨🏻‍💻 painterdrown

## 用例 & 顺序图

搜索页面 -> 搜索结果页面 -> 酒店详情页面 -> 支付页面

![](images/use_case.png)

## ECB 顺序图

![](images/ecb_seq.png)

## ECB 类图

![](images/ecb_class.png)

## 逻辑设计类图

将逻辑设计类图映射到实际项目框架的包图，用树形结构表述实现的包和类。

```
.
├── config
│   └── ...
├── controller
│   └── ReservationController
├── db
│   └── ...
├── entity
│   ├── Hotel
│   ├── Reservation
│   └── Room
├── util
│   └── ...
└── view
    ├── ChooseHotel
    ├── ConfirmReservation
    ├── SearchHotel
    └── SelectRooms
```
