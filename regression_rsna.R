library(tidyverse)
library(MASS)
library(caret)
library(glmnet)
library(xtable)


# results with shape level means and pca
run_multi = 0

# results with pooled means and pca
run_multi2 = 1

# include other shape-derived features; rotational alignment and morphometrics
include_features = 0

# set.seed(1)


args <- commandArgs(trailingOnly = TRUE)

script_dir = args[1]
cwd = script_dir
pca_path = args[2]

# cwd = getwd()
# pca_path = "pca_rsna"

df_path = file.path(cwd, pca_path,"full_df.csv")

files = list.files(path = file.path(cwd, pca_path), pattern = "merged\\.csv$", recursive = TRUE, full.names = TRUE)
files = files[grepl("vertebra_[0-9]+", files)]

df_list = lapply(files, read.csv)


conditions = c("spinal_canal_stenosis_l1_l2",	"spinal_canal_stenosis_l2_l3",
               "spinal_canal_stenosis_l3_l4",	"spinal_canal_stenosis_l4_l5",
               "spinal_canal_stenosis_l5_s1",	"left_neural_foraminal_narrowing_l1_l2",
               "left_neural_foraminal_narrowing_l2_l3", "left_neural_foraminal_narrowing_l3_l4",
               "left_neural_foraminal_narrowing_l4_l5",	"left_neural_foraminal_narrowing_l5_s1",
               "right_neural_foraminal_narrowing_l1_l2",	"right_neural_foraminal_narrowing_l2_l3",
               "right_neural_foraminal_narrowing_l3_l4",	"right_neural_foraminal_narrowing_l4_l5",
               "right_neural_foraminal_narrowing_l5_s1",	"left_subarticular_stenosis_l1_l2",
               "left_subarticular_stenosis_l2_l3",	"left_subarticular_stenosis_l3_l4",
               "left_subarticular_stenosis_l4_l5",	"left_subarticular_stenosis_l5_s1",
               "right_subarticular_stenosis_l1_l2",	"right_subarticular_stenosis_l2_l3",
               "right_subarticular_stenosis_l3_l4",	"right_subarticular_stenosis_l4_l5",
               "right_subarticular_stenosis_l5_s1")

full_df = Reduce(function(x, y) merge(x, y, by = c("study_id","series_id","series_description", conditions), all = TRUE), df_list)

full_df[] <- lapply(full_df, function(x) {
  if (is.character(x) || is.factor(x)) {
    x <- trimws(as.character(x))
    x[x == ""] <- NA
  }
  x
})


# rotations = read.csv("/home/ns13f/Documents/Spine/info/rotational_alignment_results.csv")
# rotations = read.csv("C:/Users/niths/Documents/Research/Spine/info/rotational_alignment_results.csv")
# rotations = read.csv("C:/Users/niths/Documents/Research/Spine/rotational_alignment_results.csv")

rename_map <- c(
  "_44_45" = "_l4_l5",
  "_43_44" = "_l3_l4",
  "_42_43" = "_l2_l3",
  "_41_42" = "_l1_l2"
)

# rotations_wide <- rotations %>%
#   pivot_wider(
#     id_cols = study_id,
#     names_from = vert_pair,
#     values_from = c(flexion, axial_rot, lateral_bend),
#     names_glue = "{.value}_{vert_pair}"
#   ) %>%
#   rename_with(
#     ~ str_replace_all(.x, rename_map)
#   ) %>%
#   rename(series_id = study_id)

# morph = read.csv("C:/Users/niths/Documents/Research/Spine/info/morphometrics.csv")
# morph = read.csv("C:/Users/niths/Documents/Research/Spine/morphometrics.csv")

# morph_wide <- morph %>%
#   pivot_wider(
#     id_cols = study_id,
#     names_from = vert_pair,
#     values_from = c(d_left_min, d_left_mean, d_right_min, d_right_mean, disc_min, disc_avg, disc_width, disc_length, disc_height
# ),
#     names_glue = "{.value}_{vert_pair}"
#   ) %>%
#   rename_with(
#     ~ str_replace_all(.x, rename_map)
#   ) %>%
#   rename(series_id = study_id)

# full_df = merge(full_df, rotations_wide, by = c("series_id"))
# full_df = merge(full_df, morph_wide, by = c("series_id"))

write.csv(full_df, df_path, row.names = FALSE)

dir.create(file.path(cwd, pca_path, "models"))
dir.create(file.path(cwd, pca_path, "models_allspine"))

condition_map <- c(
  "spinal_canal_stenosis" = "Spinal Canal Stenosis"
)

i_map <- c(
  "41" = "L1 Vertebra",
  "42" = "L2 Vertebra",
  "43" = "L3 Vertebra",
  "44" = "L4 Vertebra",
  "45" = "L5 Vertebra",
  "91" = "T12-L1 Disc",
  "92" = "L1-L2 Disc",
  "93" = "L2-L3 Disc",
  "94" = "L3-L4 Disc",
  "95" = "L4-L5 Disc",
  "100" = "L5-S Disc"
)

i_map_component <- c(
  "41" = "Vertebra",
  "42" = "Vertebra",
  "43" = "Vertebra",
  "44" = "Vertebra",
  "45" = "Vertebra",
  "91" = "Disc",
  "92" = "Disc",
  "93" = "Disc",
  "94" = "Disc",
  "95" = "Disc",
  "100" = "Disc",
  "291" = "Canal",
  "292" = "Canal",
  "293" = "Canal",
  "294" = "Canal",
  "295" = "Canal",
  "2100" = "Canal"
)

label_map <- c(
  "_l5_s1" = "L5-S Disc",
  "_l4_l5" = "L4-L5 Disc",
  "_l3_l4" = "L3-L4 Disc",
  "_l2_l3" = "L2-L3 Disc",
  "_l1_l2" = "L1-L2 Disc"
)

# how many components to use for regression
num_comp <- "^PC([1-9]|[1-9][0-9]|[1-4][0-9]{2}|500)_"
# num_comp <- "^PC([1-9]|[1-9][0-9]|[1-2][0-9]{2}|300)_"
# num_comp <- "^PC([1-9]|[1-9][0-9]|[1][0-9]{2}|200)_"
# num_comp = "^PC([1-9]|[1-4][0-9]|50)_"
# num_comp = "^PC([1-9]|[1-9][0-9]|100)_"
# num_comp = "^PC([1-9]|10)_"

pc_cols <- grep(num_comp, colnames(full_df), value = TRUE)


Discs = c(100,95,94,93,92,91)

IVD_list = list("_l5_s1",
                  "_l4_l5",
                  "_l3_l4",
                  "_l2_l3",
                  "_l1_l2")
# IVD_labels = list(c(100,45,2100),c(95,44,45,295),c(94,43,44,294),c(93,42,43,293),c(92,41,42,292))
IVD_labels = list(c(100,45),c(95,44,45),c(94,43,44),c(93,42,43),c(92,41,42))

dict = setNames(IVD_labels, IVD_list)

build_biomech_features <- function(df, include_products = TRUE, include_diffs = TRUE, include_squares = TRUE, include_cubes = TRUE, include_fourth = TRUE, include_fifth = TRUE, include_mag = TRUE) {
  pc_pattern <- "^PC([1-9]|[1-9][0-9]|[1-4][0-9]{2}|600)_([0-9]+)$"
  
  pc_cols <- grep(pc_pattern, names(df), value = TRUE)
  
  # Extract structure ID (41, 42, 93, etc.)
  structure_id <- sub(pc_pattern, "\\2", pc_cols)
  
  # Split columns by vertebra/disc
  groups <- split(pc_cols, structure_id)
  
  new_features <- list()
  
  for (gid in names(groups)) {
    
    vars <- groups[[gid]]
    sub_df <- df[, vars, drop = FALSE]
    
    tmp <- list()
    
    # Squared terms
    if (include_squares) {
      sq <- sub_df^2
      colnames(sq) <- paste0(vars, "_sq")
      
      tmp[["squares"]] <- sq
      
    }
    
    # Cubed terms
    if (include_cubes) {
      cub <- sub_df^3
      colnames(cub) <- paste0(vars, "_cub")
      
      tmp[["cubes"]] <- cub
      
    }
    
    if (include_fourth) {
      fourth <- sub_df^4
      colnames(fourth) <- paste0(vars, "_fourth")
      
      tmp[["fourths"]] <- fourth
      
    }
    
    if (include_fifth) {
      fif <- sub_df^5
      colnames(fif) <- paste0(vars, "_fif")
      
      tmp[["fifths"]] <- fif
      
    }
    
    # Pairwise products
    if (include_products) {
      prod_list <- combn(vars, 2, simplify = FALSE)
      prods <- lapply(prod_list, function(pair) {
        df[[pair[1]]] * df[[pair[2]]]
      })
      prod_names <- sapply(prod_list, function(pair)
        paste(pair[1], pair[2], "prod", sep = "_"))
      prods <- setNames(prods, prod_names)
      tmp[["products"]] <- as.data.frame(prods)
    }
    
    # Pairwise differences
    if (include_diffs) {
      prod_list <- combn(vars, 2, simplify = FALSE)
      diffs <- lapply(prod_list, function(pair) {
        df[[pair[1]]] - df[[pair[2]]]
      })
      diff_names <- sapply(prod_list, function(pair)
        paste(pair[1], pair[2], "diff", sep = "_"))
      diffs <- setNames(diffs, diff_names)
      tmp[["diffs"]] <- as.data.frame(diffs)
    }
    
    # L2 magnitude
    if (include_mag) {
      sub_mat <- as.matrix(sub_df)
      storage.mode(sub_mat) <- "numeric"
      
      mag <- sqrt(rowSums(sub_mat^2))
      mag_name <- paste0("PCmag_", gid)
      tmp[["mag"]] <- data.frame(mag)
      names(tmp[["mag"]]) <- mag_name
    }
    
    
    new_features[[gid]] <- do.call(cbind, tmp)
  }
  
  # Combine all new features
  new_features_df <- do.call(cbind, new_features)
  
  # Return original + engineered
  cbind(df, new_features_df)
}


if (run_multi == 1){
  # Multinomial
  
  for (condition in c(
    "spinal_canal_stenosis"
    # ,
    # "left_neural_foraminal_narrowing",
    # "right_neural_foraminal_narrowing",
    # "left_subarticular_stenosis",
    # "right_subarticular_stenosis"
  )) {
    
    pvals = list()
    conditions = list()
    components = list()
    all_pred_train <- character()
    all_true_train <- character()
    all_pred_train_stage1 <- character()
    all_true_train_stage1 <- character()
    all_pred <- character()
    all_true <- character()
    all_pred_stage1 <- character()
    all_true_stage1 <- character()
    classes <- levels(factor(full_df$spinal_canal_stenosis_l1_l2))  
    
    all_stage1_coefs = list()
    
    
    response_vars = names(full_df)[grepl(paste0("^", condition), names(full_df))]
    
    
    for (response in response_vars) {
      label = gsub(condition, "",response)
      
      predictor_vars = names(full_df)[grepl(paste0(num_comp), names(full_df))]
      
      predictor_vars = unique(predictor_vars)
      
      # print(predictor_vars)
      
      reg = full_df %>%
        dplyr::select(all_of(c(response, predictor_vars))) %>%
        drop_na() %>%
        distinct()
      
      if (length(unique(reg[,response])) <= 1) next
      
      # set.seed(123)
      
      
      size = nrow(reg)
      
      sink(file.path(cwd,pca_path,paste0("allspine_multinomial_logistic_results_", response, ".txt"))) 
      
      y = as.factor(reg[[response]])
      X = reg[,predictor_vars, drop = FALSE]
      
      # X <- build_biomech_features(X,
      #                             include_products = FALSE,
      #                             include_diffs = FALSE,
      #                             include_squares = TRUE,
      #                             include_cubes = TRUE,
      #                             include_fourth = TRUE,
      #                             include_fifth = TRUE,
      #                             include_mag = FALSE)
      
      # X <- model.matrix(~ (.)^2, data = X)[, -1]
      orig_names <- colnames(X)
      
      idx <- createDataPartition(y, p = 0.9, list = FALSE)
      
      X_train_scaled <- X[idx, , drop = FALSE]
      X_test_scaled  <- X[-idx, , drop = FALSE]
      
      y_train <- factor(y[idx])
      y_test  <- factor(y[-idx], levels = levels(y_train))
      
      class_counts <- table(y_train)
      print(class_counts)
      
      print(table(y_test))
      
      train_df <- data.frame(X_train_scaled, y = y_train)
      test_df  <- data.frame(X_test_scaled,  y = y_test)
      
      
      if (any(class_counts < 3)) {
        cat("Skipping fold — not enough samples in a class\n")
        sink()
        next
      }
      
      class_weights <- length(y_train) /
        (length(class_counts) * class_counts)
      
      obs_weights <- as.numeric(class_weights[y_train]^1)
      
      first_cat = "Normal/Mild"
      first_cat_opp = "Diseased"
      first_cat_replace = "Moderate"
      
      # ============================================================
      # STAGE 1: Normal/Mild  vs  (Moderate + Severe)
      # ============================================================
      y_stage1 <- ifelse(y_train == first_cat, first_cat, first_cat_opp)
      y_stage1 <- factor(y_stage1)
      
      y_stage1_test <- ifelse(y_test == first_cat, first_cat, first_cat_opp)
      y_stage1_test <- factor(y_stage1_test)
      
      # Stage 1 class weights (ONLY 2 classes now)
      stage1_counts <- table(y_stage1)
      stage1_weights <- length(y_stage1) / (length(stage1_counts) * stage1_counts)
      
      obs_weights_stage1 <- as.numeric(stage1_weights[y_stage1]^1)
      
      # start_time = Sys.time()
      
      cv_stage1 <- try(cv.glmnet(
        x = as.matrix(X_train_scaled),
        y = y_stage1,
        family = "binomial",
        alpha = 0.01,
        weights = obs_weights_stage1,
        type.measure = "class",
        nfolds = 3
      ), silent = TRUE)
      
      coefs = coef(cv_stage1, s = "lambda.min")
      print(coefs)
      
      coef_mat = as.matrix(coefs)
      
      coef_df_stage1 = data.frame(
        Feature = rownames(coef_mat),
        # Coefficient = coef_mat[, 1],
        row.names = NULL
      )
      
      coef_df_stage1[[response]] = coef_mat[,1]
      
      write.csv(coef_df_stage1, file.path(cwd, pca_path, "models_allspine", paste0("reg_coefs_",response,"_allspine.csv")), row.names = FALSE)
      
      if (inherits(cv_stage1, "try-error")) next
      
      # ============================================================
      # STAGE 2: Moderate vs Severe (ONLY diseased cases)
      # ============================================================
      diseased_idx <- which(y_train != first_cat)
      
      # Skip if not enough samples for stage 2
      if (length(unique(y_train[diseased_idx])) < 2 || length(diseased_idx) < 8) {
        cat("Skipping fold — not enough Moderate/Severe samples\n")
        sink()
        next
      }
      
      X_train_stage2 <- X_train_scaled[diseased_idx, , drop = FALSE]
      y_train_stage2 <- droplevels(y_train[diseased_idx])
      weights_stage2 <- obs_weights[diseased_idx]/sum(obs_weights[diseased_idx])
      
      cv_stage2 <- try(cv.glmnet(
        x = as.matrix(X_train_stage2),
        y = y_train_stage2,
        family = "binomial",
        alpha = 0.1,
        weights = weights_stage2,
        type.measure = "deviance",
        nfolds = 10
      ), silent = TRUE)
      
      if (inherits(cv_stage2, "try-error")) next
      
      # ============================================================
      # TRAINING PREDICTIONS
      # ============================================================
      pred_stage1_train <- predict(cv_stage1, as.matrix(X_train_scaled),
                                   s = "lambda.min", type = "class")
      levels_stage1 <- c(first_cat, first_cat_opp)
      
      y_pred_stage1_train = factor(pred_stage1_train, levels = levels_stage1)
      y_train_stage1 = factor(y_stage1, levels = levels_stage1)
      
      cm_stage1 = confusionMatrix(y_pred_stage1_train,y_train_stage1)
      print(cm_stage1)
      
      final_pred_train <- rep(NA, length(pred_stage1_train))
      final_pred_train[pred_stage1_train == first_cat] <- first_cat
      
      diseased_train_idx <- which(pred_stage1_train == first_cat_opp)
      
      if (length(diseased_train_idx) > 0) {
        pred_stage2_train <- predict(
          cv_stage2,
          as.matrix(X_train_scaled[diseased_train_idx, , drop = FALSE]),
          s = "lambda.min",
          type = "class"
        )
        
        # Stage 2: force any "Normal/Mild" -> "Moderate"
        pred_stage2_train <- as.character(pred_stage2_train)
        pred_stage2_train[pred_stage2_train == first_cat] <- first_cat_replace
        
        final_pred_train[diseased_train_idx] <- pred_stage2_train
      }
      
      y_pred_train <- factor(final_pred_train, levels = levels(y_train))
      cm_train <- confusionMatrix(y_pred_train, y_train)
      print(cm_train)
      print(mean(cm_train$byClass[, "Sensitivity"]))
      
      # ============================================================
      # TEST PREDICTIONS
      # ============================================================
      pred_stage1_test <- predict(cv_stage1, as.matrix(X_test_scaled),
                                  s = "lambda.min", type = "class")
      
      y_pred_test_stg1 = factor(pred_stage1_test, levels = levels_stage1)
      y_test_stg1 = factor(y_stage1_test, levels = levels_stage1)
      
      cm_stage1 = confusionMatrix(y_pred_test_stg1, y_test_stg1)
      print(cm_stage1)
      
      final_pred <- rep(NA, length(pred_stage1_test))
      final_pred[pred_stage1_test == first_cat] <- first_cat
      
      diseased_test_idx <- which(pred_stage1_test == first_cat_opp)
      
      if (length(diseased_test_idx) > 0) {
        pred_stage2_test <- predict(
          cv_stage2,
          as.matrix(X_test_scaled[diseased_test_idx, , drop = FALSE]),
          s = "lambda.min",
          type = "class"
        )
        
        # Stage 2: force any "Normal/Mild" -> "Moderate"
        pred_stage2_test <- as.character(pred_stage2_test)
        pred_stage2_test[pred_stage2_test == first_cat] <- first_cat_replace
        
        final_pred[diseased_test_idx] <- pred_stage2_test
      }
      
      y_pred <- factor(final_pred, levels = levels(y_test))
      
      cm <- confusionMatrix(y_pred, y_test)
      print(cm)
      print(mean(cm$byClass[, "Sensitivity"]))
      
      all_pred_train_stage1 = c(all_pred_train_stage1, as.character(y_pred_stage1_train))
      all_true_train_stage1 = c(all_true_train_stage1, as.character(y_train_stage1))
      
      all_pred_stage1 = c(all_pred_stage1, as.character(y_pred_test_stg1))
      all_true_stage1 = c(all_true_stage1, as.character(y_test_stg1))
      
      all_pred_train <- c(all_pred_train, as.character(y_pred_train))
      all_true_train <- c(all_true_train, as.character(y_train))
      
      all_pred <- c(all_pred, as.character(y_pred))
      all_true <- c(all_true, as.character(y_test))
      
      
      cm_stats = NULL
      
      
      sink()
    }
    
    sink(file.path(cwd,pca_path,paste0("allspine_multinomial_logistic_results_", condition, "_overall.txt"))) 
    
    all_pred_train_stage1 <- factor(all_pred_train_stage1, levels = levels_stage1)
    all_true_train_stage1 <- factor(all_true_train_stage1, levels = levels_stage1)
    
    cm_overall_train <- confusionMatrix(all_pred_train_stage1, all_true_train_stage1)
    print(cm_overall_train)
    # print(mean(cm_overall_train$byClass[, "Sensitivity"]))
    
    all_pred_train <- factor(all_pred_train, levels = classes)
    all_true_train <- factor(all_true_train, levels = classes)
    
    cm_overall_train <- confusionMatrix(all_pred_train, all_true_train)
    print(cm_overall_train)
    print(mean(cm_overall_train$byClass[, "Sensitivity"]))
    
    all_pred_stage1 <- factor(all_pred_stage1, levels = levels_stage1)
    all_true_stage1 <- factor(all_true_stage1, levels = levels_stage1)
    
    cm_overall <- confusionMatrix(all_pred_stage1, all_true_stage1)
    print(cm_overall)
    # print(mean(cm_overall$byClass[, "Sensitivity"]))
    
    all_pred <- factor(all_pred, levels = classes)
    all_true <- factor(all_true, levels = classes)
    
    cm_overall <- confusionMatrix(all_pred, all_true)
    print(cm_overall)
    print(mean(cm_overall$byClass[, "Sensitivity"]))
    
    sink()
    
  }
  
}



if (run_multi2 == 1){
  
  # Multinomial

  for (condition in c(
    "spinal_canal_stenosis",
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
    "left_subarticular_stenosis",
    "right_subarticular_stenosis"
  )) {
    
    pvals = list()
    conditions = list()
    components = list()
    all_pred_train <- character()
    all_true_train <- character()
    all_pred_train_stage1 <- character()
    all_true_train_stage1 <- character()
    all_pred <- character()
    all_true <- character()
    all_pred_stage1 <- character()
    all_true_stage1 <- character()
    classes <- levels(factor(full_df$spinal_canal_stenosis_l1_l2))  
    
    all_stage1_coefs = list()
    
    
    response_vars = names(full_df)[grepl(paste0("^", condition), names(full_df))]
    
    
    for (response in response_vars) {
          label = gsub(condition, "",response)
      
          predictor_vars = c()
        for (i in dict[[as.character(label)]]) {
          if (is.null(i)) next
          
          predictor_vars_label <- names(full_df)[grepl(paste0(num_comp,i), names(full_df))]
          predictor_vars <- c(predictor_vars, predictor_vars_label)
        
        }
          
          if (include_features == 1){
          rot_names <- c("flexion", "axial_rot", "lateral_bend")

          # Find columns that match any of the patterns AND contain the label
          rot_cols <- names(full_df)[sapply(names(full_df), function(col) {
        any(sapply(rot_names, function(p) grepl(p, col))) & grepl(label, col)
      })]

          new_rot_names <- sapply(rot_cols, function(col) {
            pat <- rot_names[sapply(rot_names, function(p) grepl(p, col))][1]

          })

          predictor_vars = c(predictor_vars, rot_cols)

          predictor_vars = unique(predictor_vars)

          morph_names <- c("d_left_min", "d_left_mean", "d_right_min", "d_right_mean", "disc_min", "disc_avg", "disc_width", "disc_length", "disc_height")

          # Find columns that match any of the patterns AND contain the label
          morph_cols <- names(full_df)[sapply(names(full_df), function(col) {
        any(sapply(morph_names, function(p) grepl(p, col))) & grepl(label, col)
      })]

          # new_rot_names <- sapply(rot_cols, function(col) {
          #   pat <- rot_names[sapply(rot_names, function(p) grepl(p, col))][1]
          #   # suffix <- ifelse(grepl(paste0(label, "$"), col), "lower", "upper")
          #   # sub(paste0("(", pat, ").*"), paste0("\\1_", suffix), col)
          # })

          predictor_vars = c(predictor_vars, morph_cols)

          predictor_vars = unique(predictor_vars)
          }
          
          # print(predictor_vars)
          
          reg = full_df %>%
            dplyr::select(all_of(c(response, predictor_vars))) %>%
            drop_na() %>%
            distinct()
          # print(nrow(reg))
          
          # names(reg)[match(rot_cols, names(reg))] <- new_rot_names
          # 
          # predictor_vars <- replace(predictor_vars, predictor_vars %in% rot_cols, new_rot_names)
          predictor_vars <- unlist(predictor_vars)
          
          
          
          # get_structure_id <- function(x) sub(".*_(\\d)$", "\\1", x)
          # get_structure_id <- function(x) sub(".*_", "", x)
          
          # structure_ids <- sapply(predictor_vars, get_structure_id)
          
          # groups <- split(predictor_vars, structure_ids)
          
        
          if (length(unique(reg[,response])) <= 1) next
          
          # set.seed(123)
          
          
          size = nrow(reg)
          
          sink(file.path(cwd,pca_path,paste0("combined_multinomial_logistic_results_", response, ".txt"))) 
          
          y = as.factor(reg[[response]])
          X = reg[,predictor_vars, drop = FALSE]
          
          # X <- build_biomech_features(X,
          #                             include_products = TRUE,
          #                             include_diffs = TRUE,
          #                             include_squares = TRUE,
          #                             include_cubes = TRUE,
          #                             include_fourth = TRUE,
          #                             include_fifth = TRUE,
          #                             include_mag = FALSE)
          
          # X <- model.matrix(~ (.)^2, data = X)[, -1]
          # X <- model.matrix(~ (.)*cluster, data = X)[, -1]
          # orig_names <- colnames(X)}, silent = TRUE)
          

          idx <- createDataPartition(y, p = 0.9, list = FALSE)
          
          X_train_scaled <- X[idx, , drop = FALSE]
          X_test_scaled  <- X[-idx, , drop = FALSE]
        
          
          y_train <- factor(y[idx])
          y_test  <- factor(y[-idx], levels = levels(y_train))


          class_counts <- table(y_train)
          print(class_counts)

          print(table(y_test))
          
          train_df <- data.frame(X_train_scaled, y = y_train)
          test_df  <- data.frame(X_test_scaled,  y = y_test)
          
          

          if (any(class_counts < 3)) {
            cat("Skipping fold — not enough samples in a class\n")
            sink()
            next
          }
          
          class_weights <- length(y_train) /
            (length(class_counts) * class_counts)

          obs_weights <- as.numeric(class_weights[y_train]^1)
          
          first_cat = "Normal/Mild"
          first_cat_opp = "Diseased"
          first_cat_replace = "Moderate"
          
          # ============================================================
          # STAGE 1: Normal/Mild  vs  (Moderate + Severe)
          # ============================================================
          y_stage1 <- ifelse(y_train == first_cat, first_cat, first_cat_opp)
          y_stage1 <- factor(y_stage1)
          
          y_stage1_test <- ifelse(y_test == first_cat, first_cat, first_cat_opp)
          y_stage1_test <- factor(y_stage1_test)
          
          # Stage 1 class weights (ONLY 2 classes now)
          stage1_counts <- table(y_stage1)
          stage1_weights <- length(y_stage1) / (length(stage1_counts) * stage1_counts)
          
          obs_weights_stage1 <- as.numeric(stage1_weights[y_stage1]^1)
          
          # start_time = Sys.time()

          cv_stage1 <- try(cv.glmnet(
            x = as.matrix(X_train_scaled),
            y = y_stage1,
            family = "binomial",
            alpha = 1,
            weights = obs_weights_stage1,
            type.measure = "class",
            nfolds = 3
          ), silent = TRUE)

          coefs = coef(cv_stage1, s = "lambda.min")
          print(coefs)
          
          coef_mat = as.matrix(coefs)
          
          coef_df_stage1 = data.frame(
            Feature = rownames(coef_mat),
            # Coefficient = coef_mat[, 1],
            row.names = NULL
          )
          
          coef_df_stage1[[response]] = coef_mat[,1]
          
          write.csv(coef_df_stage1, file.path(cwd, pca_path, "models", paste0("reg_coefs_",response,"_all.csv")), row.names = FALSE)
          
          
          if (inherits(cv_stage1, "try-error")) next
          
          # ============================================================
          # STAGE 2: Moderate vs Severe (ONLY diseased cases)
          # ============================================================
          diseased_idx <- which(y_train != first_cat)
          
          # Skip if not enough samples for stage 2
          if (length(unique(y_train[diseased_idx])) < 2 || length(diseased_idx) < 8) {
            cat("Skipping fold — not enough Moderate/Severe samples\n")
            sink()
            next
          }
          
          X_train_stage2 <- X_train_scaled[diseased_idx, , drop = FALSE]
          y_train_stage2 <- droplevels(y_train[diseased_idx])
          weights_stage2 <- obs_weights[diseased_idx]/sum(obs_weights[diseased_idx])
          
          cv_stage2 <- try(cv.glmnet(
            x = as.matrix(X_train_stage2),
            y = y_train_stage2,
            family = "binomial",
            alpha = 1,
            weights = weights_stage2,
            type.measure = "class",
            nfolds = 3
          ), silent = TRUE)
          
          if (inherits(cv_stage2, "try-error")) next
          
          # ============================================================
          # TRAINING PREDICTIONS
          # ============================================================
          pred_stage1_train <- predict(cv_stage1, as.matrix(X_train_scaled),
                                       s = "lambda.min", type = "class")
          levels_stage1 <- c(first_cat, first_cat_opp)
          
          y_pred_stage1_train = factor(pred_stage1_train, levels = levels_stage1)
          y_train_stage1 = factor(y_stage1, levels = levels_stage1)
          
          cm_stage1 = confusionMatrix(y_pred_stage1_train,y_train_stage1)
          print(cm_stage1)
          
          final_pred_train <- rep(NA, length(pred_stage1_train))
          final_pred_train[pred_stage1_train == first_cat] <- first_cat
          
          diseased_train_idx <- which(pred_stage1_train == first_cat_opp)
          
          if (length(diseased_train_idx) > 0) {
            pred_stage2_train <- predict(
              cv_stage2,
              as.matrix(X_train_scaled[diseased_train_idx, , drop = FALSE]),
              s = "lambda.min",
              type = "class"
            )
            
            # Stage 2: force any "Normal/Mild" -> "Moderate"
            pred_stage2_train <- as.character(pred_stage2_train)
            pred_stage2_train[pred_stage2_train == first_cat] <- first_cat_replace
            
            final_pred_train[diseased_train_idx] <- pred_stage2_train
          }
          
          y_pred_train <- factor(final_pred_train, levels = levels(y_train))
          cm_train <- confusionMatrix(y_pred_train, y_train)
          print(cm_train)
          print(mean(cm_train$byClass[, "Sensitivity"]))
          
          # ============================================================
          # TEST PREDICTIONS
          # ============================================================
          pred_stage1_test <- predict(cv_stage1, as.matrix(X_test_scaled),
                                      s = "lambda.min", type = "class")
          
          y_pred_test_stg1 = factor(pred_stage1_test, levels = levels_stage1)
          y_test_stg1 = factor(y_stage1_test, levels = levels_stage1)
          
          cm_stage1 = confusionMatrix(y_pred_test_stg1, y_test_stg1)
          print(cm_stage1)
          
          final_pred <- rep(NA, length(pred_stage1_test))
          final_pred[pred_stage1_test == first_cat] <- first_cat
          
          diseased_test_idx <- which(pred_stage1_test == first_cat_opp)
          
          if (length(diseased_test_idx) > 0) {
            pred_stage2_test <- predict(
              cv_stage2,
              as.matrix(X_test_scaled[diseased_test_idx, , drop = FALSE]),
              s = "lambda.min",
              type = "class"
            )
            
            # Stage 2: force any "Normal/Mild" -> "Moderate"
            pred_stage2_test <- as.character(pred_stage2_test)
            pred_stage2_test[pred_stage2_test == first_cat] <- first_cat_replace
            
            final_pred[diseased_test_idx] <- pred_stage2_test
          }
          
          y_pred <- factor(final_pred, levels = levels(y_test))
          
          cm <- confusionMatrix(y_pred, y_test)
          print(cm)
          print(mean(cm$byClass[, "Sensitivity"]))
          print(mean(cm$byClass[, "Balanced Accuracy"]))
          
          all_pred_train_stage1 = c(all_pred_train_stage1, as.character(y_pred_stage1_train))
          all_true_train_stage1 = c(all_true_train_stage1, as.character(y_train_stage1))
          
          all_pred_stage1 = c(all_pred_stage1, as.character(y_pred_test_stg1))
          all_true_stage1 = c(all_true_stage1, as.character(y_test_stg1))
          
          all_pred_train <- c(all_pred_train, as.character(y_pred_train))
          all_true_train <- c(all_true_train, as.character(y_train))
          
          all_pred <- c(all_pred, as.character(y_pred))
          all_true <- c(all_true, as.character(y_test))
          
          
          cm_stats = NULL
          
        
      sink()
    }
    
    sink(file.path(cwd,pca_path,paste0("combined_multinomial_logistic_results_", condition, "_overall.txt"))) 
    
    all_pred_train_stage1 <- factor(all_pred_train_stage1, levels = levels_stage1)
    all_true_train_stage1 <- factor(all_true_train_stage1, levels = levels_stage1)
    
    cm_overall_train <- confusionMatrix(all_pred_train_stage1, all_true_train_stage1)
    print(cm_overall_train)
    # print(mean(cm_overall_train$byClass[, "Sensitivity"]))
    
    all_pred_train <- factor(all_pred_train, levels = classes)
    all_true_train <- factor(all_true_train, levels = classes)
    
    cm_overall_train <- confusionMatrix(all_pred_train, all_true_train)
    print(cm_overall_train)
    print(mean(cm_overall_train$byClass[, "Sensitivity"]))
    print(mean(cm_overall_train$byClass[, "Balanced Accuracy"]))
    
    all_pred_stage1 <- factor(all_pred_stage1, levels = levels_stage1)
    all_true_stage1 <- factor(all_true_stage1, levels = levels_stage1)
    
    cm_overall <- confusionMatrix(all_pred_stage1, all_true_stage1)
    print(cm_overall)
    # print(mean(cm_overall$byClass[, "Sensitivity"]))
    
    all_pred <- factor(all_pred, levels = classes)
    all_true <- factor(all_true, levels = classes)
    
    cm_overall <- confusionMatrix(all_pred, all_true)
    print(cm_overall)
    print(mean(cm_overall$byClass[, "Sensitivity"]))
    print(mean(cm_overall$byClass[, "Balanced Accuracy"]))
    
    sink()
    
    
  }
  
}
