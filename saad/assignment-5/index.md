# 系统分析与设计 Assignment 5

> ⏰ 2018-04-27 21:22:59<br/>
> 👨🏻‍💻 painterdrown

## 按用例构建领域模型

![](images/lingyu.png)

## 数据库建模（E-R 模型）

![](images/db.png)

```SQL
create table consumer 
(
   name                 char(20)                       not null,
   email                char(50)                       not null,
   constraint PK_CONSUMER primary key clustered (name)
);

create table hotel 
(
   name                 char(50)                       not null,
   star                 decimal                        null,
   "index"              char(20)                       not null,
   code                 char(20)                       null,
   address              dec(50)                        null,
   constraint PK_HOTEL primary key clustered ("index")
);

create table location 
(
   code                 char(20)                       not null,
   name                 char(50)                       not null,
   constraint PK_LOCATION primary key clustered (code, name)
);

create table request 
(
   "hotel name"         char(50)                       not null,
   city                 char(20)                       null,
   "room type"          char(10)                       not null,
   "check in date"      datetime                       not null,
   "check out date"     datetime                       not null,
   hotel_index          char(20)                       null,
   con_name             char(20)                       null,
   special              char(100)                      null,
   "request index"      char(20)                       not null,
   constraint PK_REQUEST primary key clustered ("check in date", "check out date", "request index")
);

create table room 
(
   "room id"            char(10)                       not null,
   "hotel index"        char(20)                       null,
   type                 char(20)                       not null,
   "availble num"       decimal                        not null,
   constraint PK_ROOM primary key clustered ("room id")
);

create table "room description" 
(
   "room id"            char(10)                       null,
   type                 char(20)                       not null,
   "list price"         double                         not null
);

alter table hotel
   add constraint FK_HOTEL_REFERENCE_LOCATION foreign key (code, name)
      references location (code, name)
      on update restrict
      on delete restrict;

alter table request
   add constraint FK_REQUEST_REFERENCE_HOTEL foreign key (hotel_index)
      references hotel ("index")
      on update restrict
      on delete restrict;

alter table request
   add constraint FK_REQUEST_REFERENCE_CONSUMER foreign key (con_name)
      references consumer (name)
      on update restrict
      on delete restrict;

alter table room
   add constraint FK_ROOM_REFERENCE_HOTEL foreign key ("hotel index")
      references hotel ("index")
      on update restrict
      on delete restrict;

alter table "room description"
   add constraint "FK_ROOM DES_REFERENCE_ROOM" foreign key ("room id")
      references room ("room id")
      on update restrict
      on delete restrict;
```
