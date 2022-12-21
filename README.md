<!--
 * @Descripttion: 
 * @version: 
 * @Author: Adrian Lin
 * @Date: 2022-12-17 14:43:42
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2022-12-21 17:56:56
-->
### PNMTF-2D
PNMTF的二维并行版本

---

### 文件架构
- `dataset`
- `model`
- `code`
- `experiment.ini`
- `dataset.ini`

#### 1. dataset
dataset目录为：`dataset_name/processed/`，存放处理好的文件`text.pkl`和`label.pkl`
- `text.pkl`：经过预处理后的文本数据。
- `label.pkl`：每篇文档对应的`label`标签，该文件形式为一维列表（如[2,3,4,1,1,...,2,3,4]）。
代码会根据text.pkl自动生成相对于的`tfidf.pkl`和`vocab.pkl`（调用sklearn相关库生成，具体形式为`[word1, word2, word3, ..., wordn]`）。

#### 2. experiment.ini
实验参数的设置通过该文件来维护，具体设置情况可以参考`experiment.ini`，有注释说明。

#### 3. dataset.ini
`dataset.ini`维护数据集的详细情况，如词表、文档数、主题数以及数据集存放位置等。

#### 4. 运行方式
由于是并行模型，因此需要安装`mpi4py`，`python -m pip install mip4py`。注意：python版本别太高，推荐使用`3.8.5`；版本太高的话，安装`mpi4py`可能会出现问题。
- 服务器上运行示例：
  ```shell
  mpiexec -n 进程数 python PNMTF-2D-V1.py --data_name classic4 --exp_ini super1-4_PNMTF-2D-V1_CLASSIC4 --pr 进程数 --pc 进程数
  ```
  - 主要参数：  
    - `data_name`和`exp_ini`来运行程序，用来指定运行的模型和数据集；
    - `pc、pr`：列和行通信进程数。（注意：pc*pr == 总进程数（即-n指定的进程数））
- 天河服务器上运行示例：
  ```shell
  yhrun -N 8 -n 192 -p bigdata python3 -u ./PNMTF-2D-V1.py --data_name classic4 --exp_ini super1-4_PNMTF-2D-V1_CLASSIC4 --pr 32 --pc 6
  ```
  - 主要参数：  
    - `data_name`和`exp_ini`来运行程序，用来指定运行的模型和数据集；
    - `pc、pr`：列和行通信进程数；（注意：pc*pr == 总进程数（即-n指定的进程数））
    - `-N`：节点数量（注意，每个节点的cpu为24核）；
    - `-n`：总进程数（会平均分配到各个节点当中）；
    - `-p`：计算分区。