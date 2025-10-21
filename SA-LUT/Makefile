# Makefile for SA-LUT Environment Setup
# ======================================

SHELL := /bin/bash
ENV_NAME := salut_env

.PHONY: setup

setup:
	@echo "========================================="
	@echo "SA-LUT Environment Setup"
	@echo "========================================="
	@echo ""
	@echo "[1/3] Creating conda environment..."
	@conda env create -f environment.yml
	@echo ""
	@echo "[2/3] Installing quadrilinear_cpp..."
	@cd quadrilinear_cpp && bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(ENV_NAME) && python setup.py install"
	@echo ""
	@echo "[3/3] Installing trilinear_cpp..."
	@cd trilinear_cpp_torch1.11 && bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(ENV_NAME) && python setup.py install"
	@echo ""
	@echo "========================================="
	@echo "Setup complete! Activate with:"
	@echo "  conda activate $(ENV_NAME)"
	@echo "========================================="
