args <- commandArgs(trailingOnly = TRUE)

parse_arg <- function(x) {
  kv <- strsplit(x, "=")[[1]]
  setNames(list(kv[2]), kv[1])
}

arg_list <- do.call(c, lapply(args, parse_arg))

data_path       <- arg_list[["data_path"]]
out_path        <- arg_list[["out_path"]]
omicsarules_dir <- arg_list[["omicsarules_dir"]]

supp <- as.numeric(arg_list[["supp"]])
conf <- as.numeric(arg_list[["conf"]])
maxl <- as.numeric(arg_list[["maxl"]])
minl <- as.numeric(arg_list[["minl"]])

suppressMessages({
  source(file.path(omicsarules_dir, "R", "bi.data.R"))
  source(file.path(omicsarules_dir, "R", "so.rules.noGOSim.R"))
  library(outliers)
  library(arules)
})

tryCatch({

  data <- read.csv(data_path, header = TRUE, row.names = 1)
  data <- t(data)

  binary_data <- bi.data(
    file = data,
    row.g = TRUE,
    geneSel = FALSE,
    methods = "median"
  )

  rules <- so.rules.noGOSim(
    binary_data,
    data,
    symbol = TRUE,
    row.g = TRUE,
    method = "median",
    supp = supp,
    conf = conf,
    maxl = maxl,
    minl = minl
  )

  write.csv(rules, out_path, row.names = FALSE)

  quit(status = 0)

}, error = function(e) {

  cat("ERROR:", conditionMessage(e), "\n", file = stderr())

  quit(status = 1)

})
