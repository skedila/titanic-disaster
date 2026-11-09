# rocker/r-ver pins CRAN to a snapshot matching this R version,
# so package versions are reproducible
FROM rocker/r-ver:4.4.1

WORKDIR /app

COPY install_packages.R .
RUN Rscript install_packages.R

COPY titanic_model.R .

# The data folder is mounted at runtime (see README), not copied into the image
ENV DATA_DIR=/app/data

CMD ["Rscript", "titanic_model.R"]
