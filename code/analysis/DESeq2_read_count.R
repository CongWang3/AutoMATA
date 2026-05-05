# working directory
setwd("E:/deskTop/multi_omics/manu/GitHub/code/analysis/")
library(DESeq2)
library(dplyr)
library(optparse)
library(vegan)
library(ggrepel)
library(ggpubr)
library(ellipse)

option_list <- list(
  make_option(c("-i", "--expression_file"), type="character", default="../../data/analysis_example/expression_read_count.txt", action="store", help="This argument is expression file path"),
  make_option(c("-k", "--info_file"), type="character", default="../../data/analysis_example/group_info_read_count.txt", action="store", help="This argument is group information file path"),
  make_option(c("-c", "--fc_thr"), type="double", action="store", default="1", help="This argument decides log2FC threshold for differential expression analysis"),
  make_option(c("-d", "--padj_thr"), type="double", action="store", default="1", help="This argument decides padj threshold for differential expression analysis"),
  make_option(c("-e", "--correction"), type="character", action="store", default="BH", help="This argument defines hypothesis correction method, including none, BH, BY, holm, hochberg, hommel, or bonferroni")

)
opt = parse_args(OptionParser(option_list = option_list, usage = "This Script is to conduct differential expression analysis and generate volcano and cluster plots!", add_help_option=FALSE))

# Screening differential gene threshold
fc_thr <- opt$fc_thr
padj_thr <- opt$padj_thr

# Load data
# NOTE: 1. You need to make sure that the order of the samples in the expression file corresponds to the group info file order here
#       2. Keep the row names in the group file the same as the column names in the expression file: Control_1, Control_2, Treatment_1, Treatment_2
#       3. The Group file must contain a Group column, and the value of the group column must be 'Control' or 'Treatment'
counts <- read.table(
  paste("/xp/www/AutoMATA/download_data/Jobs/", opt$jobID, "/", opt$jobID, "_data.txt", sep = ""),
  header = TRUE,
  sep = "\t",
  check.names = FALSE,
  stringsAsFactors = FALSE,
  fill = TRUE,
  comment.char = "",
  quote = ""
)
group_info <- read.table(opt$info_file, header = TRUE, sep = "\t", check.names = FALSE, fill = TRUE, comment.char = "")
groups <- factor(group_info$Group, levels = c("Control", "Treatment"))

gene_names <- trimws(as.character(counts[[1]]))
valid_gene <- !(is.na(gene_names) | gene_names == "")
if (any(!valid_gene)) {
  counts <- counts[valid_gene, , drop = FALSE]
  gene_names <- gene_names[valid_gene]
  cat("Rows with empty gene names have been removed\n")
}
if (any(duplicated(gene_names))) {
  cat("Duplicate gene names detected; keeping first occurrence only\n")
  keep_idx <- !duplicated(gene_names)
  counts <- counts[keep_idx, , drop = FALSE]
  gene_names <- gene_names[keep_idx]
}
rownames(counts) <- gene_names
counts <- counts[, -1, drop = FALSE]


counts[] <- lapply(counts, function(x) as.numeric(as.character(x)))

counts <- counts[which(rowSums(counts)!=0),] # Delete the gene with an expression level of 0

# Empty value processing KNN imputation
if(any(is.na(counts))) {
  cat("begin KNN imputation\n")
  library(impute)
  counts <- impute.knn(as.matrix(counts))$data
}

# Delete rows with low expression (less than 1 read)
counts <- counts[rowMeans(counts)>1,]

# Create  DESeqDataSet object 
dds <- DESeqDataSetFromMatrix(
  countData = counts,
  colData = data.frame(group = groups),
  design = ~ group
)

# Filter low expression genes (at least 10 reads summing)
keep <- rowSums(counts(dds)) >= 10
dds <- dds[keep, ]

# Differential expression analysis
dds <- DESeq(dds, minReplicatesForReplace=5, parallel = FALSE)  

# Extract results
DESeq2_results <- results(dds, contrast = c("group", "Treatment", "Control"), pAdjustMethod = opt$correction)
DESeq2_results <- as.data.frame(DESeq2_results)


# Change column names: log2FoldChange -> logFC
DESeq2_results <- dplyr::rename(DESeq2_results, logFC = log2FoldChange)
gene <- rownames(DESeq2_results)
DESeq2_results <- cbind(gene,DESeq2_results)
# Sort the table, ascending by padj value, continuing with log2FC for the same padj value.
DESeq2_results <- DESeq2_results[order(DESeq2_results$padj, DESeq2_results$logFC, decreasing = c(FALSE, TRUE)), ]

# # Save results to file
# write.table(DESeq2_results, filename, row.names = FALSE, sep='\t')

# Screening for differential genes
# log2FC≥1 & padj<0.01, signal is up, which represents a significantly upregulated gene
# log2FC≤-1 & padj<0.01, signal is down, which represents a significantly downregulated gene
# The rest, signal is none, which represents non-distinct genes
DESeq2_results[which(DESeq2_results$logFC >= fc_thr & DESeq2_results$padj < padj_thr),'sig'] <- 'up'
DESeq2_results[which(DESeq2_results$logFC <= -fc_thr & DESeq2_results$padj < padj_thr),'sig'] <- 'down'
DESeq2_results[which(abs(DESeq2_results$logFC) <= fc_thr | DESeq2_results$padj >= padj_thr),'sig'] <- 'none'

# save all up and down genes
res1_select <- subset(DESeq2_results, sig %in% c('up', 'down'))
filename <- paste(getwd(),"/result/select_all.txt", sep="")
write.table(res1_select, file = filename, row.names = FALSE, sep='\t', quote = FALSE)


# save all up genes
res1_up <- subset(DESeq2_results, sig == 'up')
filename <- paste(getwd(),"/result/select_up.txt", sep="")
write.table(res1_up, file = filename, row.names = FALSE, sep='\t', quote = FALSE)

# save all down genes
res1_down <- subset(DESeq2_results, sig == 'down')
filename <- paste(getwd(),"/result/select_down.txt", sep="")
write.table(res1_down, file = filename,  row.names = FALSE, sep='\t', quote = FALSE)

library(ggplot2)
print("Drawing PCA Plot")
tryCatch({
 
  n_gene_for_vst <- nrow(counts(dds))
  if (is.na(n_gene_for_vst) || n_gene_for_vst < 1000) {
    vsd <- varianceStabilizingTransformation(dds, blind = TRUE)
  } else {
    vsd <- tryCatch(
      suppressWarnings(vst(dds, blind = TRUE)),
      error = function(e) {
        message("vst failed, fallback to varianceStabilizingTransformation: ", e$message)
        varianceStabilizingTransformation(dds, blind = TRUE)
      }
    )
  }
  pca_input <- t(assay(vsd))

 
  group_df <- group_info
  if (ncol(group_df) >= 1) {
    sample_col <- as.character(group_df[[1]])
    if (all(rownames(pca_input) %in% sample_col)) {
      idx <- match(rownames(pca_input), sample_col)
      group_df <- group_df[idx, , drop = FALSE]
    }
  }

  rda_result <- rda(pca_input, scale = TRUE)
  pca_summary <- summary(rda_result)
  pc1_Explained <- round(pca_summary$cont$importance[2, 1] * 100, 2)
  pc2_Explained <- round(pca_summary$cont$importance[2, 2] * 100, 2)

  coords <- data.frame(scores(rda_result, display = "sites", choices = c(1, 2)))
  coords$group <- factor(group_df$Group, levels = unique(group_df$Group))

  
  var_all <- data.frame(scores(rda_result, display = "species", choices = c(1, 2)))
  var_all$func <- rownames(var_all)
  var_all$loading <- sqrt(var_all$PC1^2 + var_all$PC2^2)
  var <- head(var_all[order(var_all$loading, decreasing = TRUE), ], 20)

  group_levels <- levels(coords$group)
  n_groups <- length(group_levels)
  base_colors <- c(
    "#1F77B4FF", "#FF7F0EFF", "#2CA02CFF",
    "#9467BDFF", "#8C564BFF", "#E377C2FF",
    "#7F7F7FFF", "#BCBD22FF", "#17BECFFF"
  )
  if (n_groups <= length(base_colors)) {
    group_colors <- base_colors[seq_len(n_groups)]
  } else {
    group_colors <- grDevices::rainbow(n_groups)
  }
  names(group_colors) <- group_levels

  group_unique <- unique(coords$group)
  oval_list <- lapply(group_unique, function(g) {
    subset <- dplyr::filter(coords, group == g)
    if (nrow(subset) < 3) return(NULL)
    cov_data <- cov(subset[, c("PC1", "PC2")])
    if (any(!is.finite(cov_data))) return(NULL)
    mean_data <- colMeans(subset[, c("PC1", "PC2")])
    oval_point <- ellipse::ellipse(cov_data, centre = mean_data, level = 0.95)
    data.frame(Group = g, oval_point)
  })
  oval_list <- Filter(Negate(is.null), oval_list)
  oval_data <- if (length(oval_list) > 0) do.call(rbind, oval_list) else data.frame(Group = factor(), PC1 = numeric(), PC2 = numeric())
  if (nrow(oval_data) > 0) {
    if ("x" %in% colnames(oval_data)) colnames(oval_data)[colnames(oval_data) == "x"] <- "PC1"
    if ("y" %in% colnames(oval_data)) colnames(oval_data)[colnames(oval_data) == "y"] <- "PC2"
  }

  p_pca <- ggplot()+
    geom_point(data = coords, aes(x = PC1, y = PC2, fill = group), size = 3, color = "transparent", shape = 21)+
    geom_segment(data = var, aes(x = 0, y = 0, xend = -1.25 * PC1, yend = 1.25 * PC2),
                 arrow = arrow(angle = 22.5, length = unit(0.25, "cm"), type = "closed")) +
    geom_text_repel(data = var, aes(x = -1.275 * PC1, y = 1.275 * PC2, label = func), size = 3.8) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey") +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey") +
    geom_path(data = oval_data, aes(x = PC1, y = PC2, group = Group, color = Group), show.legend = FALSE, linetype = "dashed") +
    geom_polygon(data = oval_data, aes(x = PC1, y = PC2, group = Group, fill = Group), alpha = 0.2) +
    scale_color_manual(values = group_colors, breaks = group_levels) +
    scale_fill_manual(values = group_colors, breaks = group_levels) +
    labs(x = paste("PC1 (", pc1_Explained, "%)", sep = ""), y = paste("PC2 (", pc2_Explained, "%)", sep = "")) +
    scale_x_continuous(limits = c(min(coords$PC1) - 1.5, max(coords$PC1) + 1.5)) +
    scale_y_continuous(limits = c(min(coords$PC2) - 1.5, max(coords$PC2) + 1.9)) +
    theme_classic2() +
    theme(
      legend.title = element_blank(),
      legend.key.size = unit(35, "pt"),
      axis.line = element_line(color = "black"),
      axis.ticks = element_blank()
    )

  result_path <- automata_job_file(opt$jobID, "result/pca")
  for (dev in c("pdf", "jpeg", "tiff", "png", "bmp", "svg")) {
    ggsave(paste(result_path, dev, sep = "."), p_pca, device = dev, width = 8.8, height = 6)
  }
}, error = function(e) {
  
  message("PCA generation skipped: ", e$message)
})
print("Drawing PCA Plot End")

# draw volcano plot

print("Begin Drawing Volcano Plot")
p1 <- ggplot(DESeq2_results, aes(x = logFC, y = -log10(padj))) + 
  annotate("rect", xmin = sort(res1_down$logFC)[1], xmax = max(res1_down$logFC), 
    ymin = -log10(padj_thr), ymax = ifelse(-log10(min(res1_down$padj)) >= -log10(min(res1_up$padj)), -log10(min(res1_down$padj)), -log10(min(res1_up$padj))), fill = "
  annotate("rect", xmin = min(res1_up$logFC), xmax = max(res1_up$logFC), 
    ymin = -log10(padj_thr), ymax = ifelse(-log10(min(res1_down$padj)) >= -log10(min(res1_up$padj)), -log10(min(res1_down$padj)), -log10(min(res1_up$padj))), fill = "
  
  annotate("text", x = (sort(res1_down$logFC)[1] + max(res1_down$logFC)) / 2, y = ifelse(-log10(min(res1_down$padj)) >= -log10(min(res1_up$padj)), -log10(min(res1_down$padj)), -log10(min(res1_up$padj)))+0.1, label = "DOWN", color = "
  annotate("text", x = (min(res1_up$logFC) + max(res1_up$logFC)) / 2, y = ifelse(-log10(min(res1_down$padj)) >= -log10(min(res1_up$padj)), -log10(min(res1_down$padj)), -log10(min(res1_up$padj)))+0.1, label = "UP", color = "
  annotate("text", x = max(res1_up$logFC)+0.1, y = -log10(0.05), label = "α = 0.05", color = "#b285b2", size = 4.5) +

  geom_vline(xintercept = 0, color = "grey60", linewidth = 0.6) +
  geom_hline(yintercept = 0, color = "grey60", linewidth = 0.6) +
  geom_hline(yintercept = -log10(0.05), linetype = "dotted", color = "#b285b2", linewidth = 0.6) +

  geom_point(data = DESeq2_results, shape = 21, color = "black", alpha = 0.1, size = 1.2, stroke = 0.7, fill = "grey60") +
  scale_x_continuous(limits = c(floor(sort(res1_down$logFC)[1]), ceiling(max(res1_up$logFC)))) +
  
  labs(x = expression(paste(log[2],"FC",sep="")), y = expression(paste(-log[10]," adj.p-value",sep=""))) +
  theme_classic(base_size = 15) + 
  theme (legend.position = "none")

result_path <- paste(getwd(),"/result/volcano", sep="")
for(dev in c("pdf", "jpeg", "tiff", "png", "bmp", "svg")){
  ggsave(paste(result_path, dev, sep = "."), p1, device = dev, width = 7.5, height = 6)
}

print("End Drawing Volcano Plot")


library(ComplexHeatmap)
library(ggplotify)
print("Begin Drawing Cluster Heatmap")
df <- counts[intersect(rownames(counts),rownames(res1_select)),]
df2<- as.matrix(df)  
col_annotation <- group_info                                               
rownames(col_annotation) <- col_annotation[,1]
col_annotation <- col_annotation[,-1,drop=FALSE]

p1 <- pheatmap(df2,
                column_split = as.factor(group_info$Group),
                color = colorRampPalette(c("purple", "white", "yellow"))(255),
                clustering_distance_rows = "euclidean",
                clustering_distance_cols = "euclidean",
                show_colnames = T,
                # show_rownames = T,
                annotation_col = col_annotation,
                annotation_colors = list(Group=c(Control='#cfc6fe',Treatment='#CCDFF1')),
                fontsize = 20,
                fontsize_col =20,  
                heatmap_legend_param = list(legend_height = unit(4, "cm"),  # Set the legend height
                                            legend_width = 0.2),  # Set the legend width
                scale = "row")  # "row", "column" and "none"
p1 <- as.ggplot(p1)
result_path <- paste(getwd(),"/result/df_cluster_heatmap", sep="")
for(dev in c("pdf", "jpeg", "tiff", "png", "bmp", "svg")){
  ggsave(paste(result_path, dev, sep = "."), p1, device = dev, width=20, height=20)
}
print("End Drawing Cluster Heatmap")

