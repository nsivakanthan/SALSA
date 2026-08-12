library(tidyverse)
library(MASS)
library(caret)
library(glmnet)
library(xtable)

# results on binary classification problems
run_conditions = 1

# results on binary pfirrmann classification
run_multi = 0

# results on multi-label pfirrmann classification
run_multi2 = 0

args <- commandArgs(trailingOnly = TRUE)

script_dir = args[1]
cwd = script_dir
pca_path = args[2]

# cwd = getwd()
# pca_path = "pca_spider"

df_path = file.path(cwd, pca_path,"full_df.csv")

if (!file.exists(df_path)) {

  files <- list.files(
    path = file.path(cwd, pca_path),
    pattern = "\\merged.csv$",
    recursive = TRUE,
    full.names = TRUE
  )

  files <- files[grepl("vertebra_[0-9]+", files)]
  print(files)

  df_list <- lapply(files, read.csv)

  merged_df <- Reduce(
    function(x, y) merge(x, y, by = c("ID", "Patient", "Sex", "Age"), all = TRUE),
    df_list
  )

  gradings <- read.csv(file.path(cwd, "info", "radiological_gradings.csv"))

  full_df <- merge(
    gradings,
    merged_df,
    by = "Patient",
    all = TRUE
  )

  write.csv(full_df, df_path, row.names = FALSE)

} else {
  message("Using existing: ", df_path)
  full_df <- read.csv(df_path)
}

dir.create(file.path(cwd, pca_path, "models"))


condition_map <- c(
  "UP.endplate"       = "Upper_Endplate",
  "LOW.endplate"      = "Lower_Endplate",
  "Spondylolisthesis" = "Spondylolisthesis",
  "Disc.herniation"   = "Disc_Herniation",
  "Disc.narrowing"    = "Disc_Narrowing",
  "Disc.bulging"      = "Disc_Bulging",
  "Pfirrman.grade"    = "Pfirrman_Grade"
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
  "100" = "Disc"
)

label_map <- c(
  "2" = "L5-S Disc",
  "3" = "L4-L5 Disc",
  "4" = "L3-L4 Disc",
  "5" = "L2-L3 Disc",
  "6" = "L1-L2 Disc",
  "7" = "T12-L1 Disc"
)

type_map <- c(
  "_t1" = "T1 MRI",
  "_t2" = "T2 MRI"
)

# num_comp <- "^PC([1-9]|[1-9][0-9]|[1][0-9]{2}|200)_"
num_comp = "^PC([1-9]|[1-9][0-9]|100)_"
# num_comp = "^PC([1-9]|[1-4][0-9]|50)_"
# num_comp = "^PC([1-9]|10)_"

pc_cols <- grep(num_comp, colnames(full_df), value = TRUE)
coefs = sub("_.*", "", pc_cols)

# coefs = unique(c("Intercept", coefs, "Sex")) 
coefs = unique(c("Intercept", coefs)) 


coef_df <- data.frame(Variable = coefs)

Discs = c(100,95,94,93,92,91)

IVD_list = seq(0,9,1)
IVD_labels = list(NULL,c(100,45),c(95,44,45),c(94,43,44),c(93,42,43),c(92,41,42),c(91,41),NULL,NULL,NULL)
dict = setNames(IVD_labels, IVD_list)


if (run_conditions == 1){
  
  pvals = list()
  conditions = list()
  components = list()
  
  for (condition in c(
    "UP.endplate",
    "LOW.endplate",
    "Spondylolisthesis",
    "Disc.herniation",
    "Disc.narrowing",
    "Disc.bulging"
  )) {
    
    all_pred_train <- character()
    all_true_train <- character()

    all_pred <- character()
    all_true <- character()

    sink(file.path(cwd,pca_path,paste0("logistic_results_", condition_map[condition], ".txt")))
    
    for (type in c("_t1", "_t2")) {
      filtered = full_df[grepl(paste0(type), full_df$ID),]
      response_vars = names(full_df)[grepl(paste0("^", condition), names(full_df))]
      
      unique_vals = as.vector(na.omit(unique(full_df$IVD.label)))
      
      for (label in unique_vals) {
          predictor_vars = c()
        for (i in dict[[as.character(label)]]) {
          if (is.null(i)) next
          predictor_vars_label <- names(filtered)[grepl(paste0(num_comp,i), names(filtered))]
          predictor_vars <- c(predictor_vars, predictor_vars_label)
        }
          
          reg = filtered %>%
            dplyr::filter(IVD.label == label) %>%
            dplyr::select(all_of(c(response_vars, predictor_vars))) %>%
            drop_na() %>%
            distinct()
          
          
          if (length(unique(reg[,condition])) <= 1) {
            next}
          
          # set.seed(123)
          
          
          size = nrow(reg)
          
          y = as.factor(reg[[response_vars]])
          X = reg[,predictor_vars, drop = FALSE]
          
          idx <- createDataPartition(y, p = 0.90, list = FALSE)
          
          X_train <- X[idx, , drop = FALSE]
          X_test  <- X[-idx, , drop = FALSE]
          
          y_train <- factor(y[idx])
          y_test  <- factor(y[-idx], levels = levels(y_train))
          
          class_counts <- table(y_train)
          print(class_counts)
          
          print(table(y_test))
          
          train_df <- data.frame(X_train, y = y_train)
          test_df  <- data.frame(X_test,  y = y_test)
          
          if (any(class_counts < 3)) {
            cat("Skipping fold — not enough samples in a class\n")
            next
          }
          
          class_weights <- length(y_train) /
            (length(class_counts) * class_counts)
          
          obs_weights <- as.numeric(class_weights[y_train]^1)
          
          cv <- try(cv.glmnet(
            x = as.matrix(X_train),
            y = y_train,
            family = "binomial",
            alpha = 0,
            weights = obs_weights,
            type.measure = "class",
            nfolds = 3,
            parallel = TRUE
          ), silent = TRUE)
          
          if (inherits(cv, "try-error")) {
            print(cv)
            next
          }
          
          coefs = coef(cv, s = "lambda.min")
          print(coefs)

          features <- paste0("PC", 1:100)

          coef_df <- data.frame(
            Feature = features
          )

          for (lab in dict[[as.character(label)]]) {

            coef_mat <- as.matrix(coefs)

            # Remove intercept
            coef_mat <- coef_mat[
              rownames(coef_mat) != "(Intercept)",
              ,
              drop = FALSE
            ]

            # Select the PCs for this label
            rows <- grepl(paste0("_", lab, "$"), rownames(coef_mat))

            features_lab <- sub(
              paste0("_", lab, "$"),
              "",
              rownames(coef_mat)[rows]
            )

            # Order PC1, PC2, ..., PC100
            ord <- order(
              as.numeric(sub("PC", "", features_lab))
            )

            coef_vec <- coef_mat[rows, 1][ord]

            coef_df[[paste0("coef_", condition_map[condition],"_", lab, type)]] <- coef_vec
          }

          write.csv(coef_df, file.path(cwd, pca_path, "models", paste0("reg_coefs_",condition_map[condition],"_",dict[[as.character(label)]][1],type,".csv")), row.names = FALSE)
          
          pred_train <- predict(cv, as.matrix(X_train),
                                       s = "lambda.min", type = "class")
          levels <- c("Normal", "Diseased")
          
          
          y_pred_train = factor(pred_train,
                                levels = c("0", "1"),
                                labels = c("Normal", "Diseased"))
          y_train = factor(y_train,
                           levels = c("0", "1"),
                           labels = c("Normal", "Diseased"))
          
          cm = confusionMatrix(y_pred_train,y_train)
          print(cm)
          
          pred_test <- predict(cv, as.matrix(X_test),
                                      s = "lambda.min", type = "class")
          
          y_pred_test = factor(pred_test,
                               levels = c("0", "1"),
                               labels = c("Normal", "Diseased"))
          y_test = factor(y_test,
                          levels = c("0", "1"),
                          labels = c("Normal", "Diseased"))
          
          cm_test = confusionMatrix(y_pred_test, y_test)
          print(cm_test)
          
          
          all_pred_train <- c(all_pred_train, as.character(y_pred_train))
          all_true_train <- c(all_true_train, as.character(y_train))
          
          all_pred <- c(all_pred, as.character(y_pred_test))
          all_true <- c(all_true, as.character(y_test))
        
      }
      
    }
    all_pred_train <- factor(all_pred_train)
    all_true_train <- factor(all_true_train)
    
    cm_overall_train <- confusionMatrix(all_pred_train, all_true_train)
    print(cm_overall_train)
    
    all_pred <- factor(all_pred)
    all_true <- factor(all_true)
    
    cm_overall <- confusionMatrix(all_pred, all_true)
    print(cm_overall)
    
    sink()
  }
  
  
  coef_vars = coef_df[, sapply(coef_df[-1,], function(col) any(col != 0))]
  write.csv(coef_vars, file.path(cwd, pca_path, "reg_coefs.csv"), row.names = FALSE)
  
}


if (run_multi == 1){
  
  pvals = list()
  conditions = list()
  components = list()
  
  for (condition in c(
    "Pfirrman.grade"
  )) {
    
    all_pred_train <- character()
    all_true_train <- character()

    all_pred <- character()
    all_true <- character()

    sink(file.path(cwd,pca_path,paste0("logistic_results_", condition_map[condition], ".txt")))
    
    for (type in c("_t1", "_t2")) {
      filtered = full_df[grepl(paste0(type), full_df$ID),]
      response_vars = names(full_df)[grepl(paste0("^", condition), names(full_df))]
      
      unique_vals = as.vector(na.omit(unique(full_df$IVD.label)))
      
      for (label in unique_vals) {
          predictor_vars = c()
        for (i in dict[[as.character(label)]]) {
          if (is.null(i)) next
          predictor_vars_label <- names(filtered)[grepl(paste0(num_comp,i), names(filtered))]
          predictor_vars <- c(predictor_vars, predictor_vars_label)
        }
          
          reg = filtered %>%
            dplyr::filter(IVD.label == label) %>%
            dplyr::select(all_of(c(response_vars, predictor_vars))) %>%
            drop_na() %>%
            distinct()
          
          
          if (length(unique(reg[,condition])) <= 1) {
            next}
          
          # set.seed(123)
          
          
          size = nrow(reg)
          
          y = as.factor(reg[[response_vars]])
          y <- factor(ifelse(as.numeric(as.character(y)) <= 2, 0, 1),
                      levels = c("0", "1"))
          X = reg[,predictor_vars, drop = FALSE]
          
          idx <- createDataPartition(y, p = 0.95, list = FALSE)
          
          X_train <- X[idx, , drop = FALSE]
          X_test  <- X[-idx, , drop = FALSE]
          
          y_train <- factor(y[idx], levels = c("0", "1"))
          y_test  <- factor(y[-idx], levels = levels(y_train))
          
          class_counts <- table(y_train)
          print(class_counts)
          
          print(table(y_test))
          
          train_df <- data.frame(X_train, y = y_train)
          test_df  <- data.frame(X_test,  y = y_test)
          
          if (any(class_counts < 3)) {
            cat("Skipping fold — not enough samples in a class\n")
            next
          }
          
          class_weights <- length(y_train) /
            (length(class_counts) * class_counts)
          
          obs_weights <- as.numeric(class_weights[y_train]^1)
          
          cv <- try(cv.glmnet(
            x = as.matrix(X_train),
            y = y_train,
            family = "binomial",
            alpha = 0,
            weights = obs_weights,
            type.measure = "class",
            nfolds = 3
          ), silent = TRUE)
          
          if (inherits(cv, "try-error")) {
            print(cv)
            next
          }
          
          coefs = coef(cv, s = "lambda.min")
          print(coefs)
        
          # Extract coefficient names at the optimal lambda
          coef_mat = as.matrix(coefs)
          
          coef_df = data.frame(
            Feature = rownames(coef_mat),
            row.names = NULL
          )
          coef_df[[response_vars]] = coef_mat[,1]
          write.csv(coef_df, file.path(cwd, pca_path, "models", paste0("reg_coefs_",condition_map[condition],"_",dict[[as.character(label)]][1],type,".csv")), row.names = FALSE)
          
          pred_train <- predict(cv, as.matrix(X_train),
                                       s = "lambda.min", type = "class")
          levels <- c("Normal", "Diseased")
          
          
          y_pred_train = factor(pred_train,
                                levels = c("0", "1"),
                                labels = c("Normal", "Diseased"))
          y_train = factor(y_train,
                           levels = c("0", "1"),
                           labels = c("Normal", "Diseased"))
          
          cm = confusionMatrix(y_pred_train,y_train)
          print(cm)
          
          pred_test <- predict(cv, as.matrix(X_test),
                                      s = "lambda.min", type = "class")
          
          y_pred_test = factor(pred_test,
                               levels = c("0", "1"),
                               labels = c("Normal", "Diseased"))
          y_test = factor(y_test,
                          levels = c("0", "1"),
                          labels = c("Normal", "Diseased"))
          
          cm_test = confusionMatrix(y_pred_test, y_test)
          print(cm_test)
          
          
          all_pred_train <- c(all_pred_train, as.character(y_pred_train))
          all_true_train <- c(all_true_train, as.character(y_train))
          
          all_pred <- c(all_pred, as.character(y_pred_test))
          all_true <- c(all_true, as.character(y_test))
        
      }
      
    }
    all_pred_train <- factor(all_pred_train)
    all_true_train <- factor(all_true_train)
    
    cm_overall_train <- confusionMatrix(all_pred_train, all_true_train)
    print(cm_overall_train)
    
    all_pred <- factor(all_pred)
    all_true <- factor(all_true)
    
    cm_overall <- confusionMatrix(all_pred, all_true)
    print(cm_overall)
    
    sink()
  }
  
  
  coef_vars = coef_df[, sapply(coef_df[-1,], function(col) any(col != 0))]
  write.csv(coef_vars, file.path(cwd, pca_path, "reg_coefs_Pfirrmann.csv"), row.names = FALSE)
  
}

if (run_multi2 == 1){
  
  pvals = list()
  conditions = list()
  components = list()
  
  for (condition in c(
    "Pfirrman.grade"
  )) {
    
    all_pred_train <- character()
    all_true_train <- character()
    
    all_pred <- character()
    all_true <- character()
    
    sink(file.path(cwd,pca_path,paste0("multi_logistic_results_", condition_map[condition], ".txt")))
    
    for (type in c("_t1", "_t2")) {
      filtered = full_df[grepl(paste0(type), full_df$ID),]
      response_vars = names(full_df)[grepl(paste0("^", condition), names(full_df))]
      
      unique_vals = as.vector(na.omit(unique(full_df$IVD.label)))
      
      for (label in unique_vals) {
        predictor_vars = c()
        for (i in dict[[as.character(label)]]) {
          if (is.null(i)) next
          predictor_vars_label <- names(filtered)[grepl(paste0(num_comp,i), names(filtered))]
          predictor_vars <- c(predictor_vars, predictor_vars_label)
        }
        
        reg = filtered %>%
          dplyr::filter(IVD.label == label) %>%
          dplyr::select(all_of(c(response_vars, predictor_vars))) %>%
          drop_na() %>%
          distinct()
        
        
        if (length(unique(reg[,condition])) <= 1) {
          next}
        
        
        size = nrow(reg)
        
        y = as.factor(reg[[response_vars]])

        X = reg[,predictor_vars, drop = FALSE]
        
        idx <- createDataPartition(y, p = 0.90, list = FALSE)
        
        X_train <- X[idx, , drop = FALSE]
        X_test  <- X[-idx, , drop = FALSE]
        
        y_train <- factor(y[idx])
        y_test  <- factor(y[-idx], levels = levels(y_train))
        
        class_counts <- table(y_train)
        print(class_counts)
        
        print(table(y_test))
        
        train_df <- data.frame(X_train, y = y_train)
        test_df  <- data.frame(X_test,  y = y_test)
        
        if (any(class_counts < 3)) {
          cat("Skipping fold — not enough samples in a class\n")
          next
        }
        
        class_weights <- length(y_train) /
          (length(class_counts) * class_counts)
        
        obs_weights <- as.numeric(class_weights[y_train]^1)
        
        cv <- try(cv.glmnet(
          x = as.matrix(X_train),
          y = y_train,
          family = "multinomial",
          alpha = 0,
          weights = obs_weights,
          type.measure = "class",
          nfolds = 3
        ), silent = TRUE)
        
        if (inherits(cv, "try-error")) {
          print(cv)
          next
        }
        
        coefs = coef(cv, s = "lambda.min")
        print(coefs)
        
        # Extract coefficient names at the optimal lambda
        coef_mat = as.matrix(coefs)

        features <- unlist(lapply(1:5, function(cls)
          paste0("PC", 1:100, "_", cls)))

        coef_df <- data.frame(
          Feature = features
        )
        

        for (lab in dict[[as.character(label)]]) {

          coef_vec <- c()

          for (cls in names(coefs)) {

            coef_mat <- as.matrix(coefs[[cls]])

            # Remove intercept
            coef_mat <- coef_mat[rownames(coef_mat) != "(Intercept)", , drop = FALSE]

            rows <- grepl(paste0("_", lab, "$"), rownames(coef_mat))

            features <- sub(paste0("_", lab, "$"), "", rownames(coef_mat)[rows])
            ord <- order(as.numeric(sub("PC", "", features)))

            coef_vec <- c(coef_vec, coef_mat[rows, 1][ord])
          }

          coef_df[[paste0("coef_Pfirrmann_", lab, type)]] <- coef_vec
        }
        
        write.csv(coef_df, file.path(cwd, pca_path, "models", paste0("reg_coefs_multi_",condition_map[condition],"_",dict[[as.character(label)]][1],type,".csv")), row.names = FALSE)
        
        pred_train <- predict(cv, as.matrix(X_train),
                              s = "lambda.min", type = "class")
        levels <- levels(y_train)
        
        
        y_pred_train = factor(pred_train,
                              levels = levels(y_train))
        y_train = factor(y_train,
                         levels = levels(y_train))
        
        cm = confusionMatrix(y_pred_train,y_train)
        print(cm)
        
        pred_test <- predict(cv, as.matrix(X_test),
                             s = "lambda.min", type = "class")
        
        y_pred_test = factor(pred_test,
                             levels = levels(y_train))
        y_test = factor(y_test,
                        levels = levels(y_train))
        
        cm_test = confusionMatrix(y_pred_test, y_test)
        print(cm_test)
        
        
        all_pred_train <- c(all_pred_train, as.character(y_pred_train))
        all_true_train <- c(all_true_train, as.character(y_train))
        
        all_pred <- c(all_pred, as.character(y_pred_test))
        all_true <- c(all_true, as.character(y_test))
        
      }
      
    }
    all_pred_train <- factor(all_pred_train)
    all_true_train <- factor(all_true_train)
    
    cm_overall_train <- confusionMatrix(all_pred_train, all_true_train)
    print(cm_overall_train)
    print(mean(cm_overall_train$byClass[, "Sensitivity"]))
    
    all_pred <- factor(all_pred)
    all_true <- factor(all_true)
    
    cm_overall <- confusionMatrix(all_pred, all_true)
    print(cm_overall)
    print(mean(cm_overall$byClass[, "Sensitivity"]))
    
    sink()
  }
  
  
  coef_vars = coef_df[, sapply(coef_df[-1,], function(col) any(col != 0))]
  write.csv(coef_vars, file.path(cwd, pca_path, "reg_coefs_Pfirrman_multi.csv"), row.names = FALSE)
  
}


