# AutoMATA

**AutoMATA: a deep learning-enhanced bioinformatics platform for multi-omics data processing, exploration and modelling**

**Platform Introduction and User Guide Video:**
visit website: [bilibili](https://www.bilibili.com/video/BV1LDdAB3EXi/?vd_source=057f1ccccc12750f57f6d28b9c852bbdhttps://www.bilibili.com/video/BV1LDdAB3EXi/?vd_source=057f1ccccc12750f57f6d28b9c852bbd) or [YouTube](https://youtu.be/5EFewqRPKoc)

## Platform Docker Usage

Access the platform via a browser. There is no need to clone the full source code to the server; only `docker-compose.prod.yml` and `.env.prod` are required. Container images are pulled from GitHub Container Registry (GHCR); the current default paths for the upstream repositories are: `ghcr.io/congwang3/automata-frontend:<tag>` and `ghcr.io/congwang3/automata-backend:<tag>`.

### 1. Download Docker Software

```bash
sudo apt-get update

sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
    
curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Offical
# $ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  
# Offical
# echo \
#   "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
#   $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# start Docker
sudo systemctl enable docker
sudo systemctl start docker

# test
docker --version
docker compose version
```

### 2. Files and Catalogues

Select a deployment root directory on the server (referred to below as `DEPLOY_DIR`, for example, `/xp/www/AutoMATA-prod`). You need to place the `docker-compose.prod.yml` and `.env.prod` files in the root directory. These two files are in GitHub.

Create a persistent directory under `DEPLOY_DIR` (we recommend creating it manually for the first deployment):

```bash
export DEPLOY_DIR=/xp/www/AutoMATA-prod

mkdir -p "$DEPLOY_DIR"
cd "$DEPLOY_DIR"

mkdir -p data/mysql data/redis data/uploaded_files data/logs \
  data/agent_byok data/download_data data/example \
  data/reference_sql data/r_library data/stringdb_cache
```

Layout illustration：

```
DEPLOY_DIR/
├── docker-compose.prod.yml
├── .env.prod
└── data/
    ├── mysql/
    ├── redis/
    ├── uploaded_files/
    ├── logs/
    ├── agent_byok/
    ├── download_data/
    ├── example/
    ├── reference_sql/
    ├── r_library/
    └── stringdb_cache/
```

### 3. Configure `.env.prod`

```bash
chmod 600 "$DEPLOY_DIR/.env.prod"
```

#### 3.1 Items that must be modified

| Variable                              | Description                                                  |
| ------------------------------------- | ------------------------------------------------------------ |
| `DB_PASSWORD` / `MYSQL_ROOT_PASSWORD` | Change to a strong password; After the first initialization, any random modifications may result in the library being unable to log in. You need to refer to the MySQL documentation for handling or rebuilding the data volume. |
| `SECRET_KEY`                          | Change to a random long string, for example: openssl rand -hex 32. |
| `CORS_ORIGINS`                        | JSON array, fill in the real front-end source, such as ["https://www.example.com"] |

#### 3.2 Items that suggest to keep the default value

| Variable    | Default value          | Description                                                  |
| ----------- | ---------------------- | ------------------------------------------------------------ |
| `DB_HOST`   | `mysql`                | Consistent with the compose service name, do not change it to IP. |
| `REDIS_URL` | `redis://redis:6379/0` | Consistent with the compose service name.                    |
| `APP_ENV`   | `production`           | It must be consistent with the backend environment in compose. |

#### 3.3 Port 

Default:`FRONTEND_HOST_PORT=8080`,`BACKEND_HOST_PORT=1800`,  `DOWNLOAD_HOST_PORT=18001`. If `.env.prod `is modified, the gateway upstream must be synchronized.

#### 3.4 Optional 

- **Reference Annotation Library**: Place the SQL files in `data/reference_sql/` and uncomment `REFERENCE_DATA_SQL_DIR=/app/reference_sql` in `.env.prod`.
- **AI Agent**：Set `AGENT_ENABLED=true` and enter the corresponding vendor API key; place the user's BYOK file in the host machine's `data/agent_byok/` directory (which has been mounted into the container).

### 4. Pull the image and start

All commands must be run in the `DEPLOY_DIR` directory and must include the `--env-file .env.prod` ; otherwise, Compose will be unable to resolve the variables in `image:`.

```bash
cd "$DEPLOY_DIR"

docker compose --env-file .env.prod -f docker-compose.prod.yml pull
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d
docker compose --env-file .env.prod -f docker-compose.prod.yml ps
```

Expected: `mysql` and `redis` are healthy; `backend` becomes healthy after running for a while; `frontend` is running.

View the backend logs: 

```
docker compose --env-file .env.prod -f docker-compose.prod.yml logs backend --tail 100
```

### 5. Verification

The port matches the one in `.env.prod` (the following is the default):

```bash
curl -sS http://127.0.0.1:18005/health
curl -sS http://127.0.0.1:18001/health
curl -sS -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:8080/
```

| Port  | Service                         |
| ----- | ------------------------------- |
| 18005 | backend API                     |
| 18001 | download service                |
| 8080  | frontend (Nginx in a container) |

If this fails, please troubleshoot first and then continue to configure the public domain name.

### 6. API Key for AI Agent

Before entering your API key, please verify that it is valid to ensure that the agent does not become unusable due to an invalid key or insufficient funds.

```bash
# Verify
# Under normal circumstances, 200 should be returned

# DeepSeek
export DEEPSEEK_KEY='API_Key' 
curl -sS 'https://api.deepseek.com/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${DEEPSEEK_KEY}" \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"ping"}],"max_tokens":8}' \
  -w "\nHTTP:%{http_code}\n"
  
# Qwen
export QWEN_KEY='DashScope_API_Key' 
curl -sS 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${QWEN_KEY}" \
  -d '{"model":"qwen-plus","messages":[{"role":"user","content":"ping"}],"max_tokens":16}' \
  -w "\nHTTP:%{http_code}\n"
```

#### 6.1 Obtain the api key of DeepSeek

1. Visit the website: [DeepSeek Official Developer Platform](https://platform.deepseek.com/api_keys)
2. Get API Key: [DeepSeek](https://platform.deepseek.com/api_keys) and [Instruction](https://api-docs.deepseek.com/zh-cn/)
3. Recharge

#### 6.2 Obtain the api key of Qwen

1. Visit the website: [Alibaba Cloud](https://account.aliyun.com/login/login.htm?oauth_callback=https%3A%2F%2Fbailian.console.aliyun.com%2F%3Fspm%3Da2c4g.11186623.0.0.1a8c4823e1E9Ba&clearRedirectCookie=1&lang=zh#/home)
2. Apply for a free quota for new users: [Free](https://help.aliyun.com/zh/model-studio/new-free-quota?spm=5176.28197581.d_index.4.145d29a4kKhJoe)
3. Get API Key: [Qwen](https://bailian.console.aliyun.com/cn-beijing?spm=a2c4g.11186623.0.0.e14c5ec6oODeIn&tab=model#/api-key) and [Instruction](https://help.aliyun.com/zh/model-studio/get-api-key?spm=5176.30275541.J_ZGek9Blx07Hclc3Ddt9dg.1.37902f3dsZh6WM&scm=20140722.S_help@@%E6%96%87%E6%A1%A3@@2712195._.ID_help@@%E6%96%87%E6%A1%A3@@2712195-RL_APIKey-LOC_2024SPHelpResult-OR_ser-PAR1_0abb793817779891180203153e1386-V_4-PAR3_o-RE_new6-P0_0-P1_0)



## Code Usage

### 1. Environment

```bash
# Installation of the environment required to train/apply the model
conda env create -f environment.yaml

# Installation of the environment required to analyze data
conda env create -f environment_R.yaml
Rscript install_packages.R
```

### 2. Model training

```bash
# Note:
# About cmd parameters
# 1. kfold: a number represents kfold. 0 means not use kfold, 3 means use 3-fold stratified cross validation
# 2. ratio: a string represents split ratio. "0" means not use split, "8:1:1" means split a dataset into train, validation and test datasets by "8:1:1", and the seperator is colon(:)
# 3. epochs: the number of training epochs
# 4. es: the number of early stopping patience. 'es' is usually smaller than epochs, you can set it to the value same as 'epochs' to disable 'es'
# 5. lr: learning rate
# 6. bs: batch size
# 7. loss_function: please input one of crossentropy, nllloss and focalloss
# 8. optimizer_function: please input one of adam, rmsprop and sgd
# 9. output_size: The number of labels. 2 for binary classification, 4 for 4-class classification.
# 10. random_seed: Keep the same random seed to ensure reproducibility.
# 11. r_method: Regularization method: l1, l2, maxnorm, sparsity, or none
# 12. r_weight: Regularization weight/strength
# 13. dropout_rate: Dropout rate is used to prevent overfitting, and its range is [0,1)
# 14. feature_method: Feature selection method: PCC, SPEARMAN, CHI2, RF.


# About model parameters
# User can change hidden_size_1 to fit the distribution of data, etc.

# About data
# The first column is the name of samples, the last column is the label of samples (label must be a integer number in order from 0), the first row is the column name (the last column must be 'Label'). The seperater is '\t'.

# Next is the command line examples to train the model
conda activate automata
cd code/train
# train AutoEncoder (using default parameters)
python autoencoder.py --kfold 0 --ratio 0 --epochs 50 --es 10 --lr 0.01 --bs 32 --loss_function crossentropy --optimizer_function adam --output_size 2 --random_seed 42 --r_method l2 --r_weight 0.1 maxnorm --dropout_rate 0.2 --feature_method PCC
# train CNN
python cnn.py --kfold 0 --ratio 8:1:1 --epochs 50 --es 10 --lr 0.01 --bs 32 --loss_function crossentropy --optimizer_function adam --output_size 4 --random_seed 3 --r_method maxnorm --r_weight 0.2 maxnorm --dropout_rate 0 --feature_method CHI2
# train LSTM
python lstm.py --kfold 4 --ratio 0 --epochs 50 --es 10 --lr 0.005 --bs 32 --loss_function nllloss --optimizer_function rmsprop --output_size 2 -–random_seed 42
# train MLP (using default parameters)
python mlp.py
```



### 3. Model application

```bash
# Note:
# About cmd parameters
# 1. bs: batch size
# 2. model_type: please input one of CNN, AutoEncoder, LSTM, MLP, RBFN, RNN and Transformer
# 3. model_path: the path of model you want to use.
# 4. model_autoencoder_path: when 'model_type' is AutoEncoder, the parameter should be specialized to the path of genertated model_autoencoder.pth

# About data
# The first column is the name of samples, the last column is the label of samples (label must be a integer number in order from), the first row is the column name (the last column must be 'Label'). The seperater is '\t'.

# Next is the command line examples to apply the model
conda activate automata
cd code/application
# apply CNN (using default parameters)
python general.py
# apply Transformer
python general.py --model_type Transformer --model_path ./model/model.pth --bs 16
# apply AutoEncoder
python general.py --model_type AutoEncoder --model_path ./model/model_cls.pth --model_autoencoder_path ./model/model_autoencoder.pth --bs 8
# apply SOM
python som.py
```



### 4. Data analysis

```bash
conda activate R_442
cd code/analysis
```

#### **correlation heatmap**

```bash
# Note:
# About cmd parameters
# -i (input): the path of input data file. The first row of input data file is the column name. The seperater is '\t'.

# cmd examples
Rscript cor_heatmap.r
Rscript cor_heatmap.r -i cor_heatmap_example.txt 
```

#### differential gene cluster heatmap

```bash
# Note:
# About cmd parameters
# -i (input): the path of input data file. The first row of input data file is the column name. The first column of input data file is the row name. The seperater is '\t'.
# -a (type): the type of output figure, decide whether need to output row/column annotations. Please input one of data_with_row_col, data_with_col_annotation, data_with_row_annotation and only_data. 'data_with_row_col' indicates that annotations for rows and columns need to be output. 'data_with_col_annotation' indicates that annotations for columns need to be output (The group/Age/Grade/Stage/Sex in the example figure). 'data_with_row_annotation' indicates that annotations for rows need to be output (The Path/Celltype in the example figure). 'only_data' indicates that no annotations are output.
# -b (show_row_name): decides whether to show row name. Please TRUE or FALSE.
# -c (show_col_name): decides whether to show column name. Please TRUE or FALSE.
# -d (clustering_dis_row): decides the method for clustering distance for rows. Please input one of correlation, euclidean, maximum, manhattan, canberra, binary and minkowski.
# -e (clustering_dis_col): decides the method for clustering distance for columns. Please input one of correlation, euclidean, maximum, manhattan, canberra, binary and minkowski.
# -g (annotation_col_file): the path of column annotation file. If 'type' parameter is 'data_with_row_col' or 'data_with_col_annotation', then 'annotation_col_file' should be specialized.
# -f (annotation_row_file): the path of row annotation file. If 'type' parameter is 'data_with_row_col' or 'data_with_row_annotation', then 'annotation_row_file' should be specialized.
# -h (scal): decides whether to center and scale data. Please input one of row, column or none. 'row' and 'column' denote scale data to row and column respectively, 'none' represent do not scale data.
# -k (group): decides whether to display data by group. Requires the column annotation file to have a group column name of group. Please input TRUE or FALSE.

# cmd examples
Rscript df_cluster_heatmap.R
Rscript df_cluster_heatmap.R -i df_gene_cluster_example.txt -a data_with_col_annotation -g df_gene_cluster_annotation_col.txt -c FALSE -b TRUE -d correlation -e correlation -h row -k TRUE
```

#### PCA

```bash
# Note:
# About cmd parameters
# -i (input): the path of input data file. The first column of the data is the group information. The seperater is '\t'.
# -c (confidence_level): confidence level
# -b (boundary): decides whether to add boundary plots, which are the top and right sub-plots of the figure. Please input TRUE or FALSE.
# -p (permanova): decides whether to add PERMANOVA analysis. Please input TRUE or FALSE.
# -m (method): the method for permanova analysis. Please input one of 'manhattan', 'euclidean', 'canberra', 'clark', 'bray', 'kulczynski', 'jaccard', 'gower', 'altGower', 'morisita', 'horn', 'mountford', 'raup', 'binomial', 'chao', 'cao', 'mahalanobis', 'chisq', 'chord', 'hellinger', 'aitchison', and 'robust.aitchison'

# cmd examples
Rscript pca.R
Rscript pca.R -i pca_example.txt -c 0.92 -b FALSE -p FALSE
```

#### VENN

```bash
# Note:
# About cmd parameters
# -i (input): the path of input data file. The first row is the column names, and each column is the information for each group. The seperater is '\t'.
# -t (type): the type of plot. Please input one of Venn, Vennpie and Barplot.

# cmd examples
Rscript venn.R
Rscript venn.R -i venn_example.txt -t Vennpie
```

#### volcano with GSEA

```bash
# Note:
# About cmd parameters
# -i (input): the path of input data file. Keep logFC, padj, gene column names of dataset consistent. The seperater is '\t'.
# -g (gmt): the path of gmt data file. If you want to conduct GSEA analysis, then specialize it to the path gmt data file, otherwise set it to 'none'.
# -a (fc_thr): the threshold of logFC. If you wish to obtain stringent results, use 1.5, 2, or other strict thresholds (|log2FC| ≥ 1.5 or ≥ 2); conversely, use 0.58 or other lenient thresholds (|log2FC| ≥ 0.58).
# -b (padj_thr): the threshold of padj (adjusted p-value). If you wish to obtain stringent results, use 0.01 or other strict thresholds (padj < 0.01); conversely, use 0.05 or other lenient thresholds (padj < 0.05).
# -c (top): the number of higher-order genes to be emphasized. The emphasized genes are in the dashed box.
# -d (top_fc_thr): the threshold of logFC for the higher-order gene. It should be more strict than 'fc_thr'
# -e (top_padj_thr): the threshold of padj for the higher-order gene. It should be more strict than 'padj_thr'
# -f (gene_sig): the list of marked genes, the separator must be comma ','

# cmd examples
Rscript volcano_gsea_padj.R
Rscript volcano_gsea_padj.R -i volcano_example.txt -g none -a 1 -b 0.05 -c 30 -d 2 -e 0.01 -f KRAS,FOSL1
```

#### GO Enrichment

```bash
# Note:
# About cmd parameters
# -i (input): the path of input data file. The first column is gene symbol, and the column name is gene. The data file only needs the gene symbol if 'type' is bubble or bar, otherwise needs to make sure that the data file has numeric data with the column name logFC to sort by the size of the value.
# -a (type): the type of figure. Please input one of bubble, bar, chord, cluster and circle.
# -b (organism): Please input one of Homo_sapiens, Mus_musculus, Bovine, Homo_sapiens and Drosophila_melanogaster.
# -c (pvalue): pvalue threshold for GO enrichment analysis. The strict pvalue threshold is 0.01 (pvalue<0.01), and lenient threshold is 0.05 (pvalue<0.05).
# -d (qvalue): qvalue threshold for GO enrichment analysis. The gold standard is 0.05.
# -e (termNum): the number of terms for each ontology to be displayed
# -g (adjust): the pvalue adjustment method for GO enrichment analysis. Please input one of holm, hochberg, hommel, bonferroni, BH, BY, fdr, none

# cmd examples
Rscript go_enrichment.R
Rscript go_enrichment.R -i go_enrichment_example.txt -a bubble -b Mus_musculus -c 0.01 -e 10 -g BH
```

#### KEGG Enrichment

```bash
# Note:
# About cmd parameters
# -i (input): the path of input data file. The first column is gene symbol, and the column name is gene. The data file only needs the gene symbol if 'type' is bubble, otherwise needs to make sure that the data file has numeric data with the column name logFC to sort by the size of the value.
# -a (type): the type of figure. Please input one of bubble, chord, and cluster.
# -b (organism): Please input one of hsa, mmu, bos, and dme.
# -c (pvalue): pvalue threshold for KEGG enrichment analysis. The strict pvalue threshold is 0.01 (pvalue<0.01), and lenient threshold is 0.05 (pvalue<0.05).
# -d (qvalue): qvalue threshold for KEGG enrichment analysis. The gold standard is 0.05.
# -e (termNum): the number of terms for each ontology to be displayed
# -g (adjust): the pvalue adjustment method for KEGG enrichment analysis. Please input one of holm, hochberg, hommel, bonferroni, BH, BY, fdr, none

# cmd examples
Rscript kegg_enrichment.R
Rscript kegg_enrichment.R -i kegg_enrichment_example.txt -a bubble -b dme -d 0.01 -e 10 -g BH
```

#### Dumbbell_Bar

```bash
# Note:
# About cmd parameters
# -i (input): the path of input data file. This file is used to draw dumbbell diagrams
# -c (data_bar): the file path of bar plot. This file is used to draw barplot diagrams. Keep the contents and name of the first column in both two files similar and same respectively.
# -a (y_label): the y label of dumbbell_bar plot
# -b (mark_fams): the terms that will be marked, which are factors in the first column of data files, the separator must be comma ','

# cmd examples
Rscript dumbbell_bar.R
Rscript dumbbell_bar.R -i dumbbell_example.txt -c dumbbell_barplot_example.txt -a number -b Notothenioid

```

#### DESeq2 for read count

```bash
# Note:
# About cmd parameters
# -i (expression_file): the path of expression file. The first column is row names, the first row is column names.
# -k (info_file): the path of group information file. You need to make sure that the order of the samples in the expression file corresponds to the group info file order here. Keep the row names in the group file the same as the column names in the expression file: Control_1, Control_2, Treatment_1, Treatment_2. The Group file must contain a Group column, and the value of the group column must be 'Control' or 'Treatment'
# -c (fc_thr): the log2FC threshold for differential expression analysis. If you wish to obtain stringent results, use 1.5, 2, or other strict thresholds (|log2FC| ≥ 1.5 or ≥ 2); conversely, use 0.58 or other lenient thresholds (|log2FC| ≥ 0.58).
# -d (padj_thr): the padj threshold for differential expression analysis. If you wish to obtain stringent results, use 0.01 or other strict thresholds (padj < 0.01); conversely, use 0.05 or other lenient thresholds (padj < 0.05).
# -e (correction): This argument defines hypothesis correction method. Please input one of none, BH, BY, holm, hochberg, hommel, or bonferroni

# cmd examples
Rscript DESeq2_read_count.R
Rscript DESeq2_read_count.R -i expression_read_count.txt -k group_info_read_count.txt -c 2 -d 0.05 -e BH
```

#### limma for fpkm

```bash
# Note:
# About cmd parameters
# -i (expression_file): the path of expression file. The first column is row names, the first row is column names.
# -k (info_file): the path of group information file. You need to make sure that the order of the samples in the expression file corresponds to the group info file order here. Keep the row names in the group file the same as the column names in the expression file: Control_1, Control_2, Treatment_1, Treatment_2. The Group file must contain a Group column, and the value of the group column must be 'Control' or 'Treatment'
# -c (fc_thr): the log2FC threshold for differential expression analysis. If you wish to obtain stringent results, use 1.5, 2, or other strict thresholds (|log2FC| ≥ 1.5 or ≥ 2); conversely, use 0.58 or other lenient thresholds (|log2FC| ≥ 0.58).
# -d (padj_thr): the padj threshold for differential expression analysis. If you wish to obtain stringent results, use 0.01 or other strict thresholds (padj < 0.01); conversely, use 0.05 or other lenient thresholds (padj < 0.05).
# -e (correction): This argument defines hypothesis correction method. Please input one of none, BH, BY, holm, hochberg, hommel, bonferroni

# cmd examples
Rscript limma_fpkm_df.R
Rscript limma_fpkm_df.R -i expression_fpkm.txt -k group_info_fpkm.txt -c 2 -d 0.05 -e BH
```

#### PPI Network

```bash
# Note:
# About cmd parameters
# -i (input): the path of input data file. The first row is column name, and the first column is gene. The file can be consist of gene symbol, ENTREZID, or ENSEMBL
# -a (type): this argument is the type of data, please input one of SYMBOL, ENTREZID, and ENSEMBL
# -b (organism): this argument is the organism, please input one of Mus_musculus, Homo_sapiens, Drosophila_melanogaster, and Bos_taurus
# -c (score_threshold): the interaction results were filtered according to the protein interaction scores
# -d (plot_type): this parameter is the type of plot, please input one of linear, kk, and stress
# -e (show_num): only genes with more than <show_num> nodes are shown in the plot

# cmd examples
Rscript ppi.R
Rscript ppi.R -i ppi_example.txt -a SYMBOL -b Homo_sapiens -c 400 -d linear -e 5
```



### 5. **Hyperparameter optimization**

1. kfold: Generally start by trying 3, 5, or 10; 5-fold cross-validation is commonly used.
2. Dataset split ratio: Common train:validation:test set split ratios are 7:1.5:1.5 or 8:1:1. More training data is generally better.
3. Batch size: Typically start by trying 32, 64, or 128. Smaller batch sizes usually provide better generalization but train slower. Larger batch sizes train faster but may overfit.
4. Learning rate: Usually start training with 0.1, 0.01, 0.001, or 0.0001. Adjust the learning rate based on the Acc-loss curves. If the training loss curve does not decrease, try increasing the learning rate; if both training and validation loss curves plateau, it might be due to too small a learning rate or model convergence.
5. Optimizer: Adam has strong adaptability and is the default optimizer; SGD requires more fine-tuned learning rate adjustment; RMSprop performs well in RNNs and certain tasks. We recommend trying Adam first, then SGD if higher accuracy is needed.
6. Loss Function: CrossEntropy is the standard loss for multi-class problems; NLLLoss is used with LogSoftmax; Focal Loss is used to address class imbalance issues. Users can choose the appropriate loss function based on the class distribution of their dataset.
7. Epochs: Number of training rounds. Users can initially set a relatively large value (e.g., 200) to observe the loss and accuracy during training and determine when overfitting or underfitting starts. Reduce epochs if overfitting occurs, and increase epochs if underfitting occurs. Rely on early stopping mechanisms to prevent overfitting.
8. Early stopping patience: Start by trying 5, 8, or 10. If the validation loss fluctuates significantly, increase the patience value.
9. Random seed: Keep the same random seed to ensure reproducibility.
10. Regularization method: L2 regularization is the most versatile option, as it smoothly limits overly large weights; L1 regularization produces sparse weights, making it suitable for scenarios with many features where automatic feature selection is desired; MaxNorm is often used in conjunction with Dropout; Sparsity regularization is used in models such as autoencoders that require sparse representations in hidden layers. We recommend starting with L2; if sparsity is required, switch to L1; use MaxNorm and Sparsity as needed.
11.   Regularization weight/strength: Typically, start by trying values such as 1e-5, 1e-4, 1e-3, and 1e-2. Smaller values (e.g., 1e-5 to 1e-4) provide mild regularization and are suitable when there is sufficient data; larger values (e.g., 1e-3 to 1e-2) can more effectively suppress overfitting, but may lead to underfitting. It is recommended to start with 1e-4 and observe the training-validation loss curve: if the training loss is low but the validation loss is high (overfitting), increase the regularization strength; if both are high (underfitting), decrease the strength.
12.  Dropout rate: Common values include 0.2, 0.3, 0.5, and so on. A lower dropout rate (such as 0.1–0.2) is suitable for small models or when the dataset is small; 0.5 is a classic choice for many fully connected layers; for convolutional layers, a lower dropout rate (such as 0.1–0.3) is typically used, or dropout is omitted entirely. It is recommended to start with 0.5. If the training set accuracy is significantly higher than the validation set (overfitting), increase the dropout rate; if the training set accuracy is also low (underfitting), decrease the dropout rate or disable it.
13. Feature selection method: PCC is suitable for linear relationships and normally distributed data; SPEARMAN is robust for nonlinear monotonic relationships and non-normal distributions; CHI2 is suitable for discrete or binary features (which must first be discretized); RF can capture nonlinear interactions, but has a higher computational cost. It is recommended to first try PCC (for continuous variables) or SPEARMAN (non-parametric) based on the data distribution. If there are many features and computational resources permit, RF typically performs better; CHI2 is commonly used in tasks involving text or discrete features.





## NOTE

AutoMATA to be free to academia, charge for industry.
