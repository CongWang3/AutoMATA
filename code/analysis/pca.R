rm(list=ls())
# working directory
setwd("E:/deskTop/multi_omics/manu/AutoMATA/code/analysis/")

library(pacman)
pacman::p_unload(pacman::p_loaded(), character.only = TRUE)

library(ggplot2) 
library(vegan) 
library(dplyr)  
library(ggrepel) 
library(ggpubr) 
library(patchwork) 
library(optparse)

option_list <- list(
  make_option(c("-i", "--input"), type="character", default="../../data/analysis_example/pca_example.txt", action="store", help="This argument is input path"),
  make_option(c("-c", "--confidence_level"), type="double", default=0.95, action="store", help="This argument is confidence level"),
  make_option(c("-b", "--boundary"), type="logical", default=TRUE, action="store", help="This argument decides whether to add boundary plot"),
  make_option(c("-p", "--permanova"), type="logical", default=TRUE, action="store", help="This argument decides whether to add PERMANOVA analysis"),
  make_option(c("-m", "--method"), type="character", default="bray", action="store", help="If permanova is TRUE, this argument is PERMANOVA method, can be 'manhattan', 'euclidean', 'canberra', 'clark', 'bray', 'kulczynski', 'jaccard', 'gower', 'altGower', 'morisita', 'horn', 'mountford', 'raup', 'binomial', 'chao', 'cao', 'mahalanobis', 'chisq', 'chord', 'hellinger', 'aitchison', or 'robust.aitchison'.")

)
opt = parse_args(OptionParser(option_list = option_list, usage = "This Script is to draw PCA!"))

# Significance star
get_stars <- function(p){
    if (is.na(p)) {  
      return("")
    }
    if (p <= 0.001) {
    return("***")
  } else if (p <= 0.01) {
    return("**")
  } else if (p <= 0.05) {
    return("*")
  } else {
    return("")
  }
}

confidence_level <- opt$confidence_level
method <- opt$method 
permanova <- opt$permanova 
boundary <- opt$boundary
dev <- "pdf"

# Read the data. The first column of the data is the group information, the column name is Group.
table_data <- read.table(opt$input, header = TRUE, sep = "\t", check.names = FALSE)


if (nrow(table_data) == 0) {
    stop("Error: The data file is empty or the format is incorrect, please check the uploaded file")
}

if (ncol(table_data) < 2) {
    stop("Error: The data file must have at least two columns (group column and at least one data column)")
}


if (colnames(table_data)[1] != "Group") {
    warning("Warning: The first column name is not 'Group', but it is still treated as group information")
}

data_cols <- dplyr::select(table_data, colnames(table_data)[-1])

if (ncol(data_cols) == 0) {
    stop("Error: There are no numeric data columns available for PCA analysis")
}

tryCatch({
    data_matrix <- as.matrix(data_cols)
    storage.mode(data_matrix) <- "numeric"

    if (any(is.na(data_matrix))) {
        na_count <- sum(is.na(data_matrix))
        warning(paste("Warning: The data contains", na_count, "NA values, which have been replaced with 0"))
        data_matrix[is.na(data_matrix)] <- 0
    }
    # Remove the columns with "all NA" or "standard deviation 0"
    data_matrix <- data_matrix[, apply(data_matrix, 2, function(x) {
      !all(is.na(x)) && sd(x, na.rm = TRUE) > 0
    }), drop = FALSE]
    
    if (ncol(data_matrix) == 0) {
        stop("Error: There are no numeric data columns available for PCA analysis")
    }
    rda_result <- rda(data_matrix, scale=T)
    pca_summary <- summary(rda_result)
}, error = function(e) {
    stop(paste("Error: PCA analysis failed -", e$message, "\\nPlease check the data format is correct (the first column is the group, the remaining columns are numeric data)"))
})

pc1_Explained <- round(pca_summary$cont$importance[2, 1]*100, 2)
pc2_Explained <- round(pca_summary$cont$importance[2, 2]*100, 2)



coords <- data.frame(scores(rda_result, display="sites", choices=c(1,2))) %>% mutate(group = table_data[[1]])
coords$group <- factor(coords$group, levels = unique(table_data[,1]))
var <- data.frame(scores(rda_result, display="species", choices=c(1,2))) %>% mutate(func = rownames(scores(rda_result, display="species")))
var$func <- factor(var$func, levels = rownames(scores(rda_result, display="species")), labels = rownames(scores(rda_result, display="species")))


if (permanova){
    nova <- adonis2(vegdist(data_matrix, method = method) ~ table_data[,1], data = table_data)  
    R2 <- round(nova$R2[1], 3)
    Pr <- round(nova$`Pr(>F)`[1], 4)
    Pr_show <- ifelse(is.na(Pr), "NA", round(Pr, 4))
    significance_stars <- get_stars(Pr)  # star
}


group_unique <- unique(coords$group)
oval_list <- lapply(group_unique, function(g){
  subset <- dplyr::filter(coords, group == g)
  if (nrow(subset) < 3) {
    return(NULL)
  }
  cov_data <- cov(subset[, c("PC1", "PC2")])
  if (any(!is.finite(cov_data))) {
    return(NULL)
  }
  mean_data <- colMeans(subset[, c("PC1", "PC2")])
  oval_point <- ellipse::ellipse(cov_data, centre = mean_data, level = confidence_level)
  data.frame(Group = g, oval_point)
})
oval_list <- Filter(Negate(is.null), oval_list)
oval_data <- if (length(oval_list) > 0) do.call(rbind, oval_list) else NULL


if (!is.null(oval_data) && nrow(oval_data) > 0) {
  if ("x" %in% colnames(oval_data)) {
    colnames(oval_data)[colnames(oval_data) == "x"] <- "PC1"
  }
  if ("y" %in% colnames(oval_data)) {
    colnames(oval_data)[colnames(oval_data) == "y"] <- "PC2"
  }
} else {
  oval_data <- data.frame(Group = factor(), PC1 = numeric(), PC2 = numeric())
}

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

p1 <- ggplot()+
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
  labs(x = paste("PC1 (", pc1_Explained, "%)",sep=""), y = paste("PC2 (", pc2_Explained, "%)",sep="")) +  # label
  scale_x_continuous(limits = c(min(coords$PC1)-1.5, max(coords$PC1)+1.5)) + 
  scale_y_continuous(limits = c(min(coords$PC2)-1.5, max(coords$PC2)+1.9)) +  
  theme_classic2() +  
  theme(legend.title = element_blank(),  
        legend.key.size = unit(35, "pt"),  
        axis.line = element_line(color = "black"),  
        axis.ticks = element_blank())  

if (permanova){
    p1 <- p1 + annotate("text", label = paste0("PERMANOVA", "\n", "R^2", " = ", R2, "\n","p = ", Pr_show, significance_stars), x = -1.7, y = 2.5)  
}

if (boundary){
    legend <- get_legend(p1) 
    legend <- as_ggplot(legend)
    p1 <- p1 + theme(legend.position = "none")  

    # The marginal distribution curve of the PC1 axis
    p2 <- ggplot(data = coords) +  
    geom_density(aes(x = PC1, fill=group), alpha = 0.2, 
                color = 'black', position = 'identity', show.legend = FALSE) +  
    scale_fill_manual(values = group_colors, breaks = group_levels) +
    scale_x_continuous(limits = c(min(coords$PC1)-1.5, max(coords$PC1)+1.5)) +  
    theme_classic() +  
    theme(legend.title = element_blank(),  
            axis.title = element_blank(),  
            axis.text = element_blank(),  
            axis.ticks = element_blank())  


    # Marginal distribution curve of the PC2 axis
    p3 <- ggplot(data = coords) +  
    geom_density(aes(x = PC2, fill=group), alpha = 0.2, 
                color = 'black', position = 'identity', show.legend = FALSE) +  
    scale_fill_manual(values = group_colors, breaks = group_levels) +
    scale_x_continuous(limits = c(min(coords$PC2)-1.5, max(coords$PC2)+1.9)) +  
    theme_classic() +  
    theme(legend.title = element_blank(), 
            axis.title = element_blank(), 
            axis.text = element_blank(),  
            axis.ticks = element_blank()) +  
    coord_flip()  


    # Custom layout structure
    design <- "224
               113
               113"
    p1 <- p1 + p2 + p3 + legend + plot_layout(design = design)  

}



# save
result_path <- paste(getwd(),"/result/pca", sep="")
for(dev in c("pdf", "jpeg", "tiff", "png", "bmp", "svg")){
  ggsave(paste(result_path, dev, sep = "."), p1, device = dev, width = 8.8, height = 6)

}
