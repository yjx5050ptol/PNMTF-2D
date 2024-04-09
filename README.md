<!--
 * @Descripttion: 
 * @version: 
 * @Author: Adrian Lin
 * @Date: 2022-12-17 14:43:42
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2022-12-21 17:56:56
-->
### PNMTF-2D
the official code for PNMTF-2D

---

### directory structure
- `dataset`
- `model`
- `code`
- `experiment.ini`
- `dataset.ini`

#### 1. dataset
dataset dir: `dataset_name/processed/`，with pre-processed files `text.pkl` and `label.pkl`
- `text.pkl`: pre-processed documents
- `label.pkl`: the corresponding labels of each document（shape as [2,3,4,1,1,...,2,3,4]）

the `tfidf.pkl` and `vocab.pkl` will be generated by our code.


#### 2. experiment.ini
the parameter settings

#### 3. dataset.ini
the required information for each dataset

#### 4. 运行方式
`mpi4py`，`python -m pip install mip4py`.
Python version recommended : `3.8.5`, some unexpected problem will happen with newer versions.
- running command：
  ```shell
  mpiexec -n n_threads python PNMTF-2D-V1.py --data_name classic4 --exp_ini super1-4_PNMTF-2D-V1_CLASSIC4 --pr n_rows --pc n_cols
  ```
  - main paras：  
    - `data_name` `exp_ini` to specify the dataset and paras
    - `pc、pr`：the numbers of col and row threads, pr * pc = p
- command on Tianhe-2：
  ```shell
  yhrun -N 8 -n 192 -p bigdata python3 -u ./PNMTF-2D-V1.py --data_name classic4 --exp_ini super1-4_PNMTF-2D-V1_CLASSIC4 --pr 32 --pc 6
  ```
  - 主要参数：  
    - `data_name` `exp_ini` to specify the dataset and paras
    - `pc、pr`：the numbers of col and row threads, pr * pc = p
    - `-N`：num of nodes；
    - `-n`：num of threads；
    - `-p`：computation region。