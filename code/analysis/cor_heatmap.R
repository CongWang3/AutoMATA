rm(list=ls())
# working directory
setwd("E:/deskTop/multi_omics/manu/AutoMATA/code/analysis/")
library(linkET)
library(ggplot2)
library(RColorBrewer)
library(ggnewscale)
library(optparse)  

# cmd: 
# command line parameter
option_list <- list(
  make_option(c("-i", "--input"), type="character", default="../../data/analysis_example/cor_heatmap_example.txt", action="store", help="This argument is the path of input file.")
)
opt = parse_args(OptionParser(option_list = option_list, usage = "This Script is to draw correlation heatmap!"))

# load data
# NOTE: the input file should be a tab-separated file with no row names, you can change column names by yourself.
data <- read.table(opt$input, header = TRUE, sep = "\t", check.names = FALSE)

# Filter non-numeric columns (retain numeric columns)
data <- data[, sapply(data, is.numeric), drop = FALSE]


p_base <- qcorrplot(correlate(data), 
          grid_col = "grey50",
          grid_size = 0.25,
          type = "upper", 
          diag = FALSE) +
  geom_square()


sample_n <- nrow(data)
if (sample_n < 10) {
  # Small sample: Only use the basic heat map
  p1 <- p_base
} else {
  # Large sample: basic heat map + significant mark
  p1 <- p_base +
    geom_mark(size = 4,
              only_mark = TRUE,
              sig_level = c(0.05, 0.01, 0.001),
              sig_thres = 0.05,
              colour = "white")
}


p1 <- p1 +
  scale_fill_gradientn(limits = c(-1,1),  # scale_fill_gradientn
                       breaks = seq(-1,1,0.5),
                       colors = rev(brewer.pal(11, "PiYG"))) + # palette
  scale_size_manual(values = c(0.5, 1.5, 3)) +
  guides(fill = guide_colorbar(title = "Pearson's r", 
                keyheight = unit(2.2, "cm"),
                keywidth = unit(0.5, "cm"),
                order = 3)) + 
  theme(legend.box.spacing = unit(0, "pt"))


# Enter p1 in the terminal/console to view the picture
result_path <- paste(getwd(),"/result/Corr_heatmap", sep="")
for(dev in c("pdf", "jpeg", "tiff", "png", "bmp", "svg")){
  ggsave(paste(result_path, dev, sep = "."), plot=last_plot(), device = dev, width = 8.8, height = 6)
}


library(pacman)
pacman::p_unload(pacman::p_loaded(), character.only = TRUE)