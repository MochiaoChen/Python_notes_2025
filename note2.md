Python L2：循环（loops）

本质上，循环 (loops) 是一种重复执行某件事的方式。

首先，在终端窗口 (terminal window) 中输入 `code cat.py` 来创建一个新文件。

在文本编辑器中，输入以下代码：

```python
print("meow")
print("meow")
print("meow")
```

通过输入 `python cat.py` 运行此代码，你会注意到程序叫了三次 "meow"。

作为一名程序员，你需要思考如何改进那些重复输入相同内容的代码区域。想象一下，如果你想让程序 "meow" 500 次，一遍又一遍地输入 `print("meow")` 合理吗？

循环 (Loops) 能让你创建一个可以反复执行的代码块。

## While 循环 (While Loops)

`while` 循环 (while loop) 在几乎所有编程语言中都非常普遍。
这种循环会一遍又一遍地重复执行一个代码块。

在文本编辑器窗口中，将你的代码修改如下：

```python
i = 3
while i != 0:
    print("meow")
```

注意，尽管这段代码会多次执行 `print("meow")`，但它永远不会停止！它会无限循环下去。`while` 循环的工作方式是反复询问循环的条件是否满足。在这个例子中，解释器 (interpreter) 会问：“i 是否不等于 0？”。当你陷入一个无限循环时，可以按 `Ctrl+C` 来跳出循环。

为了修复这个无限循环，我们可以这样修改代码：

```python
i = 3
while i != 0:
  print("meow")
  i = i - 1
```

注意，现在我们的代码可以正常执行了，每次循环“迭代 (iteration)”时，`i` 的值都会减 1。“迭代 (iteration)”在编程中具有特殊意义，它指的是循环的一次完整周期。第一次迭代是循环的“第 0 次”迭代，第二次是“第 1 次”迭代。在编程中，我们从 0 开始计数，然后是 1，接着是 2。

我们可以进一步改进代码：

```python
i = 1
while i <= 3:
    print("meow")
    i = i + 1
```

注意，当我们写 `i = i + 1` 时，我们是将右边的值赋给左边的 `i`。在上面的代码中，我们让 `i` 从 1 开始，就像大多数人习惯的计数方式（1, 2, 3）。如果你执行上面的代码，你会看到它叫了三次 "meow"。在编程中，最佳实践是从 0 开始计数。

我们可以改进代码，让它从 0 开始计数：

```python
i = 0
while i < 3:
    print("meow")
    i += 1
```

注意，将运算符改为 `i < 3` 使得我们的代码能按预期工作。我们从 0 开始计数，它会迭代三次循环，产生三次 "meow"。另外，`i += 1` 和 `i = i + 1` 是等价的。

这个循环的逻辑流程如下：
*   开始 (start)
*   `i` 被初始化为 0 (`i = 0`)
*   检查条件 `i < 3` 是否为 `True`
*   如果是 `True`，则执行 `print("meow")`
*   执行 `i += 1`
*   回到条件检查
*   当 `i` 达到 3 时，`i < 3` 的结果为 `False`
*   循环停止 (stop)

## For 循环 (For Loops)

`for` 循环 (for loop) 是另一种类型的循环。
要最好地理解 `for` 循环，最好先从 Python 中一种叫做 `list`（列表）的新变量类型开始讲起。就像我们生活中的购物清单、待办事项清单等一样。

`for` 循环会遍历一个列表中的所有项目。例如，在文本编辑器窗口中，修改你的 `cat.py` 代码如下：

```python
for i in [0, 1, 2]:
    print("meow")
```

注意，与之前的 `while` 循环代码相比，这段代码是多么整洁。在这段代码中，`i` 从 0 开始，程序叫一声 "meow"，然后 `i` 被赋值为 1，再叫一声 "meow"，最后 `i` 被赋值为 2，叫完最后一声 "meow" 后循环结束。

虽然这段代码完成了我们想要的功能，但在处理极端情况时，仍有改进的空间。如果想迭代一百万次呢？最好是创建能处理这种极端情况的代码。因此，我们可以改进代码如下：

```python
for i in range(3):
    print("meow")
```

注意，`range(3)` 会自动提供三个值（0, 1, 和 2）。这段代码会按预期执行，产生三次 "meow"。

我们的代码还可以进一步改进。注意，我们从未在代码中明确使用 `i`。也就是说，虽然 Python 需要 `i` 来存储循环迭代的次数，但我们没有将它用于任何其他目的。在 Python 中，如果这样的变量在我们的代码中没有其他意义，我们可以简单地用一个下划线 `_` 来表示它。因此，你可以修改代码如下：

```python
for _ in range(3):
    print("meow")
```

注意，将 `i` 改为 `_` 对程序的功能没有任何影响。

我们的代码还可以进一步改进。为了拓展你对 Python 可能性的认识，考虑以下代码：

```python
print("meow" * 3)
```

注意，它会叫三次 "meow"，但程序会产生 `meowmeowmeow` 这样的结果。思考一下：你如何在每次 "meow" 之后创建一个换行符？

确实，你可以这样编辑你的代码：

```python
print("meow\n" * 3, end="")
```

注意，这段代码产生了三声 "meow"，每声都在一个新行上。通过添加 `end=""` 和 `\n`，我们告诉解释器在每次 "meow" 的结尾添加一个换行符。

## 通过用户输入改进 (Improving with User Input)

也许我们想从用户那里获取输入。我们可以使用循环来验证用户的输入。
在 Python 中，一个常见的范式是使用 `while` 循环来验证用户的输入。

例如，让我们尝试提示用户输入一个大于等于 0 的数字：

```python
while True:
    n = int(input("What's n? "))
    if n < 0:
        continue
    else:
        break
```

注意，我们引入了两个新的 Python 关键字：`continue` 和 `break`。`continue` 明确告诉 Python 进入循环的下一次迭代。而 `break` 则告诉 Python 提前“跳出”循环，而不是等待所有迭代完成。在这个例子中，当 `n` 小于 0 时，我们会继续循环的下一次迭代——最终会再次提示用户输入 "What's n?"。但是，如果 `n` 大于或等于 0，我们就会跳出循环，让程序的其余部分运行。

事实证明，在这种情况下，`continue` 关键字是多余的。我们可以改进代码如下：

```python
while True:
    n = int(input("What's n? "))
    if n > 0:
        break

for _ in range(n):
    print("meow")
```

注意，这个 `while` 循环将一直运行（永远），直到 `n` 大于 0。当 `n` 大于 0 时，循环中断。

结合我们之前学到的知识，我们可以使用函数来进一步改进我们的代码：

```python
def main():
    number = get_number()
    meow(number)

def get_number():
    while True:
        n = int(input("What's n? "))
        if n > 0:
            return n

def meow(n):
    for _ in range(n):
        print("meow")

main()
```

注意，我们不仅将代码改为了多个函数来操作，还使用了 `return` 语句将 `n` 的值返回给 `main` 函数。

## 关于列表的更多内容 (More About Lists)

想象一下著名的哈利波特世界中的霍格沃茨。
在终端中，输入 `code hogwarts.py`。

在文本编辑器中，编码如下：

```python
students = ["Hermione", "Harry", "Ron"]

print(students[0])
print(students[1])
print(students[2])
```

注意，我们有一个包含学生姓名的列表。然后我们打印出位于第 0 个位置的学生 "Hermione"。其他每个学生也都被打印出来了。

正如我们之前所展示的，我们可以使用循环来遍历列表。你可以改进你的代码如下：

```python
students = ["Hermione", "Harry", "Ron"]

for student in students:
    print(student)
```

注意，对于 `students` 列表中的每个 `student`，它都会按预期打印出学生的名字。你可能想知道为什么我们没有像之前讨论的那样使用 `_`。我们选择不这样做，因为 `student` 在我们的代码中被明确地使用了。

> 你可以在 [Python 关于列表 (lists) 的官方文档](https://docs.python.org/3/tutorial/datastructures.html)中了解更多。

## 长度 (Length)

我们可以利用 `len` 来检查名为 `students` 的列表的长度。

想象一下，你不仅想打印出学生的名字，还想打印出他们在列表中的位置。为了实现这一点，你可以这样编辑你的代码：

```python
students = ["Hermione", "Harry", "Ron"]

for i in range(len(students)):
    print(i + 1, students[i])
```

注意，执行这段代码不仅可以使用 `i + 1` 得到每个学生的位置加一，还会打印出每个学生的名字。`len` 让你能够动态地看到学生列表的长度，无论它增长了多少。

> 你可以在 [Python 关于 len 的官方文档](https://docs.python.org/3/library/functions.html#len)中了解更多。

## 字典 (Dictionaries)

`dicts` 或 `dictionaries`（字典）是一种数据结构，它允许你将**键 (keys)**与**值 (values)**关联起来。
列表是一系列值的集合，而字典则是将键与值配对。

考虑到霍格沃茨的学院，我们可能会将特定的学生分配到特定的学院。

我们可以只用列表来完成这个任务：

```python
students = ["Hermione", "Harry", "Ron", "Draco"]
houses = ["Gryffindor", "Gryffindor", "Gryffindor", "Slytherin"]
```

我们可以保证始终保持这些列表的顺序。`students` 列表中第一个位置的个体与 `houses` 列表中第一个位置的学院相关联，依此类推。然而，随着列表的增长，这会变得相当麻烦！

我们可以使用字典来改进我们的代码：

```python
students = {
    "Hermione": "Gryffindor",
    "Harry": "Gryffindor",
    "Ron": "Gryffindor",
    "Draco": "Slytherin",
}
print(students["Hermione"])
print(students["Harry"])
print(students["Ron"])
print(students["Draco"])
```

注意，我们使用 `{}` 花括号来创建一个字典。列表使用数字来索引，而字典允许我们使用单词（字符串）作为键。

运行你的代码，确保你的输出如下：
```bash
$ python hogwarts.py
Gryffindor
Gryffindor
Gryffindor
Slytherin
```

我们可以像这样改进我们的代码：

```python
students = {
    "Hermione": "Gryffindor",
    "Harry": "Gryffindor",
    "Ron": "Gryffindor",
    "Draco": "Slytherin",
}
for student in students:
    print(student)
```
执行这段代码时，`for` 循环只会遍历所有的**键 (keys)**，结果是打印出学生的名字列表。我们如何才能同时打印出值和键呢？

修改你的代码如下：

```python
students = {
    "Hermione": "Gryffindor",
    "Harry": "Gryffindor",
    "Ron": "Gryffindor",
    "Draco": "Slytherin",
}
for student in students:
    print(student, students[student])
```
注意，`students[student]` 会找到每个学生（键）对应的学院（值）。执行你的代码，你会发现输出有点乱。

我们可以通过改进 `print` 函数来清理输出：

```python
students = {
    "Hermione": "Gryffindor",
    "Harry": "Gryffindor",
    "Ron": "Gryffindor",
    "Draco": "Slytherin",
}
for student in students:
    print(student, students[student], sep=", ")
```
注意，这会在打印的每个项目之间创建一个干净的 `, ` 分隔。

如果你执行 `python hogwarts.py`，你应该看到以下内容：
```bash
$ python hogwarts.py
Hermione, Gryffindor
Harry, Gryffindor
Ron, Gryffindor
Draco, Slytherin
```

如果我们有更多关于学生的信息怎么办？我们如何将更多的数据与每个学生关联起来？

你可以想象想要将大量数据与多个键关联起来。像这样增强你的代码：

```python
students = [
    {"name": "Hermione", "house": "Gryffindor", "patronus": "Otter"},
    {"name": "Harry", "house": "Gryffindor", "patronus": "Stag"},
    {"name": "Ron", "house": "Gryffindor", "patronus": "Jack Russell terrier"},
    {"name": "Draco", "house": "Slytherin", "patronus": None},
]
```

注意，这段代码创建了一个**字典列表 (list of dicts)**。名为 `students` 的列表内部有四个字典：每个学生一个。另外，注意 Python 有一个特殊的 `None` 值，用于表示某个键没有关联的值。

现在，你可以访问关于这些学生的大量有趣数据。现在，进一步修改你的代码如下：

```python
students = [
    {"name": "Hermione", "house": "Gryffindor", "patronus": "Otter"},
    {"name": "Harry", "house": "Gryffindor", "patronus": "Stag"},
    {"name": "Ron", "house": "Gryffindor", "patronus": "Jack Russell terrier"},
    {"name": "Draco", "house": "Slytherin", "patronus": None},
]

for student in students:
    print(student["name"], student["house"], student["patronus"], sep=", ")
```

注意，`for` 循环将遍历名为 `students` 的列表中的每个字典。

> 你可以在 [Python 关于字典 (dicts) 的官方文档](https://docs.python.org/3/tutorial/datastructures.html#dictionaries)中了解更多。

## 马里奥 (Mario)

还记得经典游戏《马里奥》中英雄跳过砖块的场景吗？让我们来创建一个这个游戏的文本表示。

开始编码如下：

```python
print("#")
print("#")
print("#")
```

注意我们是如何一遍又一遍地复制粘贴相同的代码。

考虑我们如何能更好地改进代码：

```python
for _ in range(3):
    print("#")
```

注意，这基本上完成了我们想要创建的效果。

考虑一下：我们能否进一步**抽象 (abstract)** 这段代码，以便稍后解决更复杂的问题？修改你的代码如下：

```python
def main():
    print_column(3)

def print_column(height):
    for _ in range(height):
        print("#")

main()
```

注意，我们的列可以根据需要增长，而无需任何**硬编码 (hard coding)**。

现在，让我们尝试水平打印一行。修改你的代码如下：

```python
def main():
    print_row(4)

def print_row(width):
    print("?" * width)

main()
```

注意，我们现在有了可以创建从左到右的砖块的代码。

马里奥游戏中有行也有列的砖块。
考虑一下：我们如何在代码中实现行和列？修改你的代码如下：

```python
def main():
    print_square(3)

def print_square(size):

    # For each row in square (对于方块中的每一行)
    for i in range(size):

        # For each brick in row (对于行中的每一块砖)
        for j in range(size):

            #  Print brick (打印砖块)
            print("#", end="")

        # Print blank line (打印一个空行，用于换行)
        print()

main()
```

注意，我们有一个处理方块中每一行的**外层循环 (outer loop)**。然后，我们有一个在每一行中打印砖块的**内层循环 (inner loop)**。最后，我们有一个打印空行的 `print` 语句。

我们可以进一步抽象我们的代码：

```python
def main():
    print_square(3)

def print_square(size):
    for i in range(size):
        print_row(size)

def print_row(width):
    print("#" * width)

main()
```

## 总结 (Summing Up)

现在，你不断增长的 Python 技能库中又增添了一项强大的能力。在这次讲座中，我们讨论了...

*   循环 (Loops)
*   `while` 循环
*   `for` 循环
*   `len` 函数
*   列表 (`list`)
*   字典 (`dict`)
