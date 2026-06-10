VENV_DIR := ~/jupyter-env
PYTHON   := python3
PIP      := $(VENV_DIR)/bin/pip
JUPYTER  := $(VENV_DIR)/bin/jupyter

.PHONY: all install venv packages launch lab clean help

all: install

## Create virtual environment
venv:
	$(PYTHON) -m venv $(VENV_DIR)

## Install Jupyter and common data science packages
install: venv
	$(PIP) install --upgrade pip
	$(PIP) install jupyter jupyterlab numpy pandas matplotlib scikit-learn clickhouse-connect

## Install only extra packages (if venv already exists)
packages:
	$(PIP) install numpy pandas matplotlib scikit-learn

## Launch Jupyter Notebook
launch:
	$(JUPYTER) notebook

## Launch JupyterLab
lab:
	$(JUPYTER) lab

## Remove the virtual environment
clean:
	rm -rf $(VENV_DIR)

## Show available targets
help:
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "  install   Create venv and install Jupyter + data science packages"
	@echo "  launch    Start Jupyter Notebook"
	@echo "  lab       Start JupyterLab"
	@echo "  packages  Install extra packages into existing venv"
	@echo "  clean     Remove the virtual environment"
	@echo "  help      Show this message"
	@echo ""
