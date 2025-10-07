# Python L0：创建你的第一个程序 (Creating Code with Python)

`VS Code` 是一个文本编辑器 (text editor)。除了编辑文本，你还可以可视化地浏览文件并在终端 (terminal) 中运行基于文本的命令。

在终端 (terminal) 中，你可以执行 `code hello.py` 来创建一个名为 `hello.py` 的文件并开始编码。

在上面的文本编辑器中，你可以输入 `print("hello, world")`。这是一个非常经典的标准程序，几乎所有程序员在学习过程中都会编写它。

```python
print("hello, world")
```

在终端窗口 (terminal window) 中，你可以执行命令。要运行这个程序，你需要将光标移动到屏幕底部，点击终端窗口，然后输入 `python hello.py` 并按回车键。

计算机实际上只理解 0 和 1。因此，当你运行 `python hello.py` 时，Python 解释器 (interpreter) 会解析你在 `hello.py` 中创建的文本，并将其翻译成计算机可以理解的 0 和 1。

运行 `python hello.py` 程序的结果是：
```
hello, world
```
恭喜！你刚刚创建了你的第一个程序。

## 函数 (Functions)

函数 (Functions) 是计算机或编程语言已经知道如何执行的动词或动作。

在你写的 `hello.py` 程序中，`print` 函数 (function) 就知道如何在终端窗口中打印内容。

`print` 函数接收参数 (arguments)。在这个例子中，`"hello, world"` 就是传递给 `print` 函数的参数。

## 漏洞 (Bugs)

漏洞 (Bugs) 是编码过程中的自然产物。它们是你需要解决的错误和问题。不要因此气馁，这是成为一名优秀程序员过程的一部分。

想象一下，在我们的 `hello.py` 程序中，我们不小心输入了 `print("hello, world"`，忘记了 `print` 函数需要的最后一个 `)`。如果你犯了这个错误，解释器 (interpreter) 将会在终端窗口输出一条错误信息！

错误信息通常能告诉你犯了什么错，并提供如何修复的线索。但很多时候，解释器 (interpreter) 可能没有这么友好。

## 改进你的第一个 Python 程序 (Improving Your First Python Program)

我们可以让我们的第一个 Python 程序更具个性化。

在 `hello.py` 的文本编辑器中，我们可以添加另一个函数。`input` 是一个函数，它接收一个提示 (prompt) 作为参数 (argument)。我们可以这样修改代码：

```python
input("What's your name? ")
print("hello, world")
```

然而，仅仅这样修改，程序还不能输出用户输入的内容。为此，我们需要引入变量 (variables)。

## 变量 (Variables)

变量 (variable) 就像是程序中用来存放值的容器。

你可以在程序中引入自己的变量，将代码修改为：

```python
name = input("What's your name? ")
print("hello, world")
```

注意，`name = input("What's your name? ")` 中间的等号 `=` 在编程中扮演着一个特殊的角色。这个等号的作用是把右边的值**赋值 (assigns)**给左边的变量。因此，`input("What's your name? ")` 返回的值被赋值给了 `name`。

如果你像下面这样修改代码，你会得到一个意想不到的结果：

```python
name = input("What's your name? ")
print("hello, name")
```

无论用户输入什么，程序总会在终端窗口返回 `hello, name`。

进一步修改我们的代码：

```python
name = input("What's your name? ")
print("hello,")
print(name)
```

终端窗口的结果会是：
```
What's your name? David
hello
David
```
我们离我们想要的结果更近了！

> 你可以在 Python 关于[数据类型 (data types) 的官方文档 (documentation)](https://docs.python.org/3/library/stdtypes.html)中了解更多。

## 注释 (Comments)

注释 (Comments) 是程序员用来记录他们在程序中做了什么，甚至告知他人某段代码意图的方式。简而言之，它们是为你自己和看你代码的其他人留下的笔记！

你可以在程序中添加注释，以便了解程序的功能。你可以这样修改代码：

```python
# 询问用户的名字
# Ask the user for their name
name = input("What's your name? ")
print("hello,")
print(name)
```

注释也可以作为你的待办事项列表。

## 伪代码 (Pseudocode)

伪代码 (Pseudocode) 是一种重要的注释，它是一种特殊的待办事项列表，尤其在你不知道如何完成一项编码任务时非常有用。例如：

```python
# 询问用户的名字
# Ask the user for their name
name = input("What's your name? ")

# 打印 hello
# Print hello
print("hello,")

# 打印输入的名字
# Print the name inputted
print(name)
```

## 进一步改进你的第一个 Python 程序 (Further Improving Your First Python Program)

我们可以进一步修改代码：

```python
# 询问用户的名字
name = input("What's your name? ")

# 打印 hello 和输入的名字
print("hello, " + name)
```

事实证明，一些函数可以接收多个参数 (arguments)。我们可以使用逗号 `,` 来传递多个参数：

```python
# 询问用户的名字
name = input("What's your name? ")

# 打印 hello 和输入的名字
print("hello,", name)
```

如果在终端中输入 "David"，输出将会是 `hello, David`。成功了！

## 字符串 (Strings) 和参数 (Parameters)

字符串 (string)，在 Python 中被称为 `str`，是文本序列。

回顾一下我们之前的代码，它有一个视觉上的副作用，即结果会显示在多行上：

```python
# 询问用户的名字
name = input("What's your name? ")
print("hello,")
print(name)
```

函数会接收影响其行为的参数 (arguments)。如果我们查看 `print` 函数的文档 (documentation)，我们会发现 `print` 函数会自动包含一个名为 `end` 的参数，其默认值为 `end='\n'`。这个 `\n` 表示 `print` 函数在运行时会自动创建一个换行符。

然而，我们可以自己为 `end` 提供一个参数，从而不创建新行：

```python
# 询问用户的名字
name = input("What's your name? ")
print("hello, ", end="")
print(name)
```

通过提供 `end=""`，我们覆盖了 `end` 的默认值，使得第一个 `print` 语句后不会创建新行。如果输入的名字是 "David"，终端窗口的输出将是 `hello, David`。

因此，形参 (Parameters) 就是可以被函数接收的那些参数 (arguments)。

> 你可以在 Python 关于 [`print` 的官方文档](https://docs.python.org/3/library/functions.html#print)中了解更多。

### 关于引号的一个小问题

注意，在字符串中添加引号是具有挑战性的。
`print("hello,"friend"")` 将无法工作，解释器 (interpreter) 会抛出错误。

通常有两种方法来解决这个问题。第一种，你可以简单地将外面的双引号改成单引号。
另一种更常用的方法是使用反斜杠 (backslash) `\` 进行转义：`print("hello, \"friend\"")`。反斜杠告诉解释器，后面的字符应被视为字符串中的一个普通引号，从而避免解释器错误。

## 格式化字符串 (Formatting Strings)

使用字符串最优雅的方式可能是 `f-string`：

```python
# 询问用户的名字
name = input("What's your name? ")
print(f"hello, {name}")
```

注意 `print(f"hello, {name}")` 中的 `f`。这个 `f` 是一个特殊指示符，告诉 Python 以一种特殊的方式处理这个字符串，这种方式被称为 **f-字符串 (f-string)**。你会发现，在后续课程中你会非常频繁地使用这种风格的字符串。

## 关于字符串的更多内容 (More on Strings)

你永远不应该期望你的用户会按预期进行操作。因此，你需要确保用户的输入得到修正或检查。

事实证明，字符串 (string) 内置了移除空白字符 (whitespace) 的功能。

通过在 `name` 上使用 `strip()` 方法 (method)，你可以移除用户输入内容左右两边的所有空白字符。你可以这样修改代码：

```python
# 询问用户的名字
name = input("What's your name? ")

# 从 str 中移除空白字符
name = name.strip()

# 打印输出
print(f"hello, {name}")
```

现在，无论你在名字前后输入多少空格，程序都会去除它们。

使用 `title()` 方法 (method)，可以将用户名字的每个单词首字母大写：

```python
# 询问用户的名字
name = input("What's your name? ")

# 从 str 中移除空白字符
name = name.strip()

# 将每个单词的首字母大写
name = name.title()

# 打印输出
print(f"hello, {name}")
```

> **小提示**：你可能已经厌倦了在终端窗口中反复输入 `python hello.py`。你可以使用键盘上的**上箭头**键来调出最近输入的终端命令。

我们可以将代码修改得更高效，通过**方法链式调用 (method chaining)** 实现：

```python
# 询问用户的名字，移除空白字符并将每个单词的首字母大写
name = input("What's your name? ").strip().title()

# 打印输出
print(f"hello, {name}")
```

> 你可以在 Python 关于 [`str` 的官方文档](https://docs.python.org/3/library/stdtypes.html#string-methods)中了解更多关于字符串的方法。

## 整数 (Integers or int)

在 Python 中，整数 (integer) 被称为 `int`。

在数学世界中，我们熟悉 `+`, `-`, `*`, `/` 和 `%` 这些运算符 (operators)。最后一个 `%` 或模运算符 (modulo operator) 可能对你来说不太熟悉。

> **小提示**：你并非必须在文本编辑器中运行 Python 代码。在你的终端中，可以直接运行 `python` 命令进入交互模式 (interactive mode)。你会看到 `>>>` 提示符，你可以在这里运行实时的、交互式的代码，比如输入 `1+1` 并回车。

让我们创建一个新文件 `calculator.py`。首先，我们可以声明几个变量：

```python
x = 1
y = 2

z = x + y

print(z)
```
运行 `python calculator.py`，我们自然会得到结果 `3`。我们可以使用 `input` 函数使其更具交互性：

```python
x = input("What's x? ")
y = input("What's y? ")

z = x + y

print(z)
```
运行这个程序，输入 `1` 和 `2`，我们发现输出是 `12`。为什么会这样？

因为 `input` 函数返回的值是字符串 (string) 类型，`+` 号会**连接 (concatenates)** 两个字符串。因此，我们需要将这个输入从字符串转换为整数。我们可以这样做：

```python
x = input("What's x? ")
y = input("What's y? ")

z = int(x) + int(y)

print(z)
```
现在结果正确了。`int(x)` 的使用被称为“**类型转换 (casting)**”，即一个值被临时从一种变量类型（这里是字符串）转换为另一种（这里是整数）。

我们可以进一步改进我们的程序：

```python
x = int(input("What's x? "))
y = int(input("What's y? "))

print(x + y)
```
这表明你可以将函数嵌套调用。内部的函数会先运行，然后是外部的函数。

> 你可以在 Python 关于 [`int` 的官方文档](https://docs.python.org/3/library/functions.html#int)中了解更多。

## 可读性至上 (Readability Wins)

在决定如何完成一个编码任务时，请记住，对于同一个问题，多种方法都可能合理。

无论你采取何种编程方法，请记住你的代码必须具有**可读性 (readable)**。你应该使用注释 (comments) 来为你自己和他人提供关于代码功能的线索。此外，你应该以一种易于阅读的方式编写代码。

## 浮点数基础 (Float Basics)

浮点数 (floating point value / float) 是一个带有小数点的实数，例如 `0.52`。

你可以修改你的代码以支持浮点数：

```python
x = float(input("What's x? "))
y = float(input("What's y? "))

print(x + y)
```

现在用户可以输入 `1.2` 和 `3.4`，得到总数 `4.6`。

如果我们想将总数四舍五入到最近的整数，可以使用 `round()` 函数：

```python
x = float(input("What's x? "))
y = float(input("What's y? "))

z = round(x + y)

print(z)
```

如果我们想格式化长数字的输出，例如，希望看到 `1,000` 而不是 `1000`，可以使用 `f-string`：

```python
x = float(input("What's x? "))
y = float(input("What's y? "))

z = round(x + y)

# 打印格式化后的结果
print(f"{z:,}")
```
`print(f"{z:,}")` 会让输出的 `z` 在需要时包含千位分隔符。

## 关于浮点数的更多内容 (More on Floats)

我们如何对浮点数进行舍入？首先，修改你的代码如下：

```python
x = float(input("What's x? "))
y = float(input("What's y? "))

z = x / y

print(z)
```

当 `x` 输入 `2`，`y` 输入 `3` 时，结果 `z` 是 `0.6666666666666666`。

如果我们想将这个结果四舍五入到小数点后两位，可以这样做：

```python
x = float(input("What's x? "))
y = float(input("What's y? "))

z = round(x / y, 2)

print(z)
```

我们也可以使用 `f-string` 来格式化输出，达到同样的效果：

```python
x = float(input("What's x? "))
y = float(input("What's y? "))

z = x / y

# 打印保留两位小数的结果
print(f"{z:.2f}")
```
这段神秘的 `f-string` 代码会显示与我们之前的舍入策略相同的结果。

> 你可以在 Python 关于 [`float` 的官方文档](https://docs.python.org/3/library/functions.html#float)中了解更多。

## 定义自己的函数 (Def)

如果我们能创建自己的函数，那岂不是很好？

让我们回到 `hello.py` 文件，清空所有代码，从头开始：

```python
# 定义我们自己的函数
def hello(to="world"):
    print("hello,", to)

# 使用我们的函数输出
name = input("What's your name? ")
hello(name)

# 不传递参数时，使用默认值输出
hello()
```
我们使用 `def` 关键字创建了自己的 `hello` 函数。

-   注意 `def hello():` 下面的所有内容都是**缩进 (indented)** 的。Python 是一种依赖缩进的语言。它使用缩进来理解哪些代码属于上面的函数。
-   `def hello(to):` 表示这个函数接收一个名为 `to` 的形参 (parameter)。当你调用 `hello(name)` 时，计算机会将 `name` 的值传递给函数内的 `to` 变量。
-   `def hello(to="world"):` 表示我们为形参 `to` 设置了一个**默认值 (default value)**。如果在调用 `hello()` 时没有提供参数，它将使用默认值 `"world"`。

我们通常会将代码组织成一个主函数 (main function)：

```python
def main():
    name = input("What's your name? ")
    hello(name)
    hello()

# 定义我们自己的函数
def hello(to="world"):
    print("hello,", to)

main()
```
`main()` 函数是程序的入口点。代码的最后一行 `main()` 是在**调用 (calling)** `main` 函数，从而启动我们的程序。

## 返回值 (Returning Values)

在很多情况下，你不仅希望函数执行一个动作，还希望它能将一个值返回给调用它的地方。我们将这种“传回”的值称为**返回值 (return value)**。

回到我们的 `calculator.py` 文件，用以下代码重写它：

```python
def main():
    x = int(input("What's x? "))
    print("x squared is", square(x))

def square(n):
    return n * n

main()
```
在这里，`x` 的值被传递给 `square` 函数。然后，`n * n` 的计算结果通过 `return` 关键字返回到 `main` 函数中，并被 `print` 函数打印出来。

## 总结 (Summing Up)

通过这一节课的学习，你已经掌握了将在自己的程序中无数次使用的技能。你学到了：

*   在 Python 中创建你的第一个程序 (Creating your first programs)
*   函数 (Functions)
*   漏洞 (Bugs)
*   变量 (Variables)
*   注释 (Comments)
*   伪代码 (Pseudocode)
*   字符串 (Strings)
*   参数 (Parameters)
*   格式化字符串 (Formatted Strings)
*   整数 (Integers)
*   可读性原则 (Principles of readability)
*   浮点数 (Floats)
*   创建你自己的函数 (Creating your own functions)
*   返回值 (Return values)
