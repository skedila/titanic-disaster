# Installs the R packages needed by titanic_model.R.
# The rocker/r-ver:4.4.1 base image points to a dated CRAN snapshot,
# so these resolve to fixed, reproducible versions.
packages <- c("readr", "dplyr")

install.packages(packages)

# Fail the Docker build if anything did not install
missing <- packages[!packages %in% rownames(installed.packages())]
if (length(missing) > 0) {
  stop(paste("Failed to install:", paste(missing, collapse = ", ")))
}
cat("All packages installed:", paste(packages, collapse = ", "), "\n")
