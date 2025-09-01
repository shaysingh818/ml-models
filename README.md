# ML Models

This repository contains a collection of machine learning models trained on a variety of standard datasets. It serves as a centralized resource for experimenting with regression and classification tasks, showcasing model parameters, metadata, and performance across different problem domains.

## ⚙️ Getting Started

### 🧰 Prerequisites

- [Rust](https://www.rust-lang.org/tools/install)
- [Cargo](https://doc.rust-lang.org/cargo/)
- [Parquet support in Rust](https://docs.rs/parquet/latest/parquet/)

### 🚀 Run an Experiment

You can run any of the experiments via `cargo run --bin <module_name>`. For example:

```bash
cargo run --bin iris
cargo run --bin house_prices
```

# 📊 Datasets

| Dataset               | Task Type              | File Path                          |
| --------------------- | ---------------------- | ---------------------------------- |
| Iris                  | Classification         | `data/iris.parquet`                |
| Breast Cancer         | Classification         | `data/breast_cancer.parquet`       |
| California Housing    | Regression             | `data/california_housing.parquet`  |
| Diabetes              | Regression             | `data/diabetes.parquet`            |
| Student Performance   | Regression             | `data/student_performance.parquet` |
| Bike Rentals (Hourly) | Regression/Time Series | `data/bike_rentals_hourly.parquet` |
