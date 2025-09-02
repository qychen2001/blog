---
title: "SQLAlchemy 2.0 类型检查实践"
description: "从误报开始，找到SQLAlchemy 2.0 类型检查最佳实践"
date: 2025-09-02 00:00:00+0000
math: true
categories:
  - "开发"
tags:
  - "后端"
  - "SQLAlchemy"
---

## 问题的起因

写 Python 代码的时候添加类型提示是一个非常好的习惯，因为这样可以让 IDE 做类型检查，避免一些低级错误，同时在协作的时候也可以提高效率，因此我要求团队所有的工程项目都需要做到这一点。

但是当我使用 SQLAlchemy 作为 ORM 框架的时候，经常会遇到一些 IDE 的误报。这里放一个小例子：

```python
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50))
    age = Column(Integer)


engine = create_engine("sqlite:///:memory:", echo=False)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()


def generate_new_user() -> User:
    user = User(name="Tom", age=30)
    session.add(user)
    session.commit()
    return user


def print_user_info(user: User):
    user_name: str = user.name
    user_age: int = user.age
    print("type of user name:", type(user.name))
    print("type of user age:", type(user.age))
    print(f"Name: {user_name}, Age: {user_age}")


user1 = generate_new_user()

print_user_info(user1)

```

输出如下：

```
type of user name: <class 'str'>
type of user age: <class 'int'>
Name: Tom, Age: 30
```

这段代码从输出来看，类型和结果都没错。但是使用 VSCode 搭配 Pylance 作为类型检查的时候却会提示类型错误：

```
Type "Column[str]" is not assignable to declared type "str"
```

## 尝试解决

最简单的办法就是直接在行末添加 `# type: ignore` 或者使用 `getattr`。然而这样就没法发挥类型提示的作用了。显然不是最好的办法。

经过一番搜索，我找到了一篇发布于 23 年的[博客](https://changchen.me/blog/20230503/python-type-hinting-stubs/)。简单来说就是通过安装一个 `sqlalchemy-stubs` 插件，使得类型检查器优先通过 stub 中定义的接口类型解析。

看起来挺好的，试了一下也有用，但是切换到实际项目中，由于 `sqlalchemy-stubs` 是一个最后更新于 21 年的包，使得很多 SQLAlchemy 当中的新类型无法使用，例如 `UUID`。

这就让人非常难受了，难道类型检查和新类型只能二选一了吗！

## 最佳实践

然后我就看到了 SQLAlchemy 官方发布的一个博客，讲解了如何在 2.0 版本中实现完整的类型检查。

原文：[What’s New in SQLAlchemy 2.0?](ttps://docs.sqlalchemy.org/en/20/changelog/whatsnew_20.html#orm-declarative-models)

为了给 SQLAlchemy 2.0 添加完整的类型检查，需要有三步。把原文翻译如下：

* 第一步：`declarative_base()` 已被 `DeclarativeBase` 取代。在 Python 类型注解中观察到的一个限制是，似乎无法让一个由函数动态生成的类被类型检查工具识别为新类的基类。为了在不使用插件的情况下解决这个问题，可以用 `DeclarativeBase` 类来替代通常对 `declarative_base()` 的调用。`DeclarativeBase` 类会像往常一样生成相同的 `Base` 对象，不同的是类型检查工具能够识别它。

* 第二步：用 `mapped_column()` 替换对 `Column` 的声明式使用。`mapped_column()` 是一种支持 ORM 类型的构造，可直接替代 `Column` 的使用。此时，各个列尚未使用 Python 类型进行类型标注，而是被标注为 `Mapped[Any]`。

- 第三步：根据需要使用 `Mapped` 应用精确的 Python 类型。对于所有需要精确类型的属性都可以这样做；那些可以保留为 `Any` 类型的属性可以跳过。

把上面的例子进行修改如下：

```python
from sqlalchemy import Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50))
    age: Mapped[int] = mapped_column(Integer)


engine = create_engine("sqlite:///:memory:", echo=False)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()


def generate_new_user() -> User:
    user = User(name="Tom", age=30)
    session.add(user)
    session.commit()
    return user


def print_user_info(user: User):
    user_name: str = user.name
    user_age: int = user.age
    print("type of user name:", type(user_name))
    print("type of user age:", type(user_age))
    print(f"Name: {user_name}, Age: {user_age}")


user1 = generate_new_user()

print_user_info(user1)

```

至此，已经修复了 SQLAlchemy 2.0 当中关于类型注解的所有问题。似乎现在 Vibe Coding 工具尚未掌握这个技能，看来以后 PO 层还是要手改一遍。