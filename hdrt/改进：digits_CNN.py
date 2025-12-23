import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from PIL import Image

# ====== 数据集路径 ======
DATASET_PATH = "Img"
# =======================

# 配置参数
IMAGE_SIZE = (32, 32)  # 32x32的灰度图，兼顾细节和计算量
NUM_CHANNELS = 1  # 灰度图，通道数为1（彩色图是3）

# 1. 加载数据集
def load_custom_dataset(path, img_size):
    X = []
    y = []
    class_names = []
    
    for class_idx, class_name in enumerate(os.listdir(path)):
        class_path = os.path.join(path, class_name)
        if not os.path.isdir(class_path):
            continue
        class_names.append(class_name)
        
        # 只处理常见图片格式
        img_extensions = ('.png')
        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith(img_extensions):
                continue
            img_path = os.path.join(class_path, img_name)
            
            try:
                # 读取并预处理图片
                img = Image.open(img_path).convert('L')  # 灰度图
                img = img.resize(img_size)  # 统一尺寸
                img_array = np.array(img)
                
                # 优化1：二值化（去掉灰度，只留纯黑/纯白，减少噪点干扰）
                img_array = (img_array < 127).astype(np.uint8) * 255  # 灰度<127→黑(0)，≥127→白(255)
                X.append(img_array)
                y.append(class_idx)
            except Exception as e:
                print(f"跳过损坏图片 {img_path}：{e}")
                continue
    
    # 转换为numpy数组
    X = np.array(X)
    y = np.array(y)
    if len(X) == 0:
        raise ValueError("数据集为空！检查路径是否正确")
    
    # 优化2：添加通道维度（CNN需要：(样本数, 高, 宽, 通道数)）
    X = np.expand_dims(X, axis=-1)  # 形状从 (N, 32, 32) → (N, 32, 32, 1)
    # 优化3：像素归一化（从0-255→0-1，符合CNN的输入习惯）
    X = X / 255.0
    
    return X, y, class_names

# 2. 加载数据并打印信息
print(f"加载数据集：{DATASET_PATH}")
X, y, class_names = load_custom_dataset(DATASET_PATH, IMAGE_SIZE)
num_classes = len(class_names)
print(f"加载完成：{X.shape[0]}张图，{num_classes}个类别")
print(f"数据形状：{X.shape}（样本数，高度，宽度，通道数）")
print(f"像素值范围：{np.min(X)} ~ {np.max(X)}")

# 3. 处理标签（转为独热编码，适配CNN分类）
y_one_hot = to_categorical(y, num_classes=num_classes)  # 比如标签3→[0,0,0,1,0,...]

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y_one_hot, test_size=0.3, random_state=42, stratify=y_one_hot if num_classes>1 else None
)
# 保留原始标签
y_train_original = np.argmax(y_train, axis=1)
y_test_original = np.argmax(y_test, axis=1)

# 5. 构建CNN模型
model = Sequential([
    # 卷积层1：提取基础边缘特征（32个3x3卷积核）
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], NUM_CHANNELS)),
    MaxPooling2D((2, 2)),  # 池化层：降维，保留关键特征
    
    # 卷积层2：提取更复杂的形状特征（64个3x3卷积核）
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # 卷积层3：进一步提取精细特征（128个3x3卷积核）
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # 扁平化：将二维特征转为一维
    Flatten(),
    Dropout(0.5),  # 防止过拟合：随机丢弃50%的神经元
    # 全连接层：分类
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')  # 输出层：对应类别数，输出概率
])

# 编译模型
model.compile(
    optimizer='adam',  # 优化器：自适应学习率，训练更稳定
    loss='categorical_crossentropy',  # 损失函数：多分类问题
    metrics=['accuracy']  # 评估指标：准确率
)

# 打印模型结构
model.summary()

# 6. 训练模型
history = model.fit(
    X_train, y_train,
    epochs=120,  # 训练轮数
    batch_size=32,  # 批次大小
    validation_split=0.1,  # 用10%的训练集做验证，监控过拟合
    verbose=1
)

# 7. 评估模型
print("\n=== CNN evaluation ===")
# 预测测试集
y_pred = model.predict(X_test, verbose=0)
y_pred_original = np.argmax(y_pred, axis=1)  # 转回原始标签

# 计算准确率
accuracy = accuracy_score(y_test_original, y_pred_original)
print(f"测试集准确率：{accuracy:.4f}")

# 分类报告
print("\n分类报告：")
print(classification_report(y_test_original, y_pred_original, target_names=class_names, zero_division=0))

# 8. 可视化训练过程（查看准确率/损失变化）
plt.figure(figsize=(12, 4))
# 准确率曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('训练/验证准确率')
plt.xlabel('轮数')
plt.ylabel('准确率')
plt.legend()

# 损失曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('训练/验证损失')
plt.xlabel('轮数')
plt.ylabel('损失')
plt.legend()
plt.tight_layout()
plt.show()

# 9. 混淆矩阵可视化
cm = confusion_matrix(y_test_original, y_pred_original)
fig_size = max(10, num_classes*0.8)
plt.figure(figsize=(fig_size, fig_size*0.8))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title(f'CNN混淆矩阵（尺寸：{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}）', fontsize=16)
plt.colorbar()
plt.xticks(np.arange(num_classes), class_names, rotation=45, fontsize=9)
plt.yticks(np.arange(num_classes), class_names, fontsize=9)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i,j], ha="center", va="center",
                 color="white" if cm[i,j]>cm.max()/2 else "black")
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.tight_layout()
plt.show()

# 10. 可视化预测结果
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    if i >= len(X_test):
        ax.axis('off')
        continue
    # 显示图片（去掉通道维度）
    img = X_test[i].reshape(IMAGE_SIZE)
    ax.imshow(img, cmap='binary')
    # 真实标签和预测标签
    true_label = class_names[y_test_original[i]]
    pred_label = class_names[y_pred_original[i]]
    color = 'green' if true_label==pred_label else 'red'
    ax.text(0.05, 0.05, f'T:{true_label}\nP:{pred_label}',
            transform=ax.transAxes, color=color, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
plt.suptitle('CNN预测结果展示', fontsize=16)
plt.tight_layout()
plt.show()