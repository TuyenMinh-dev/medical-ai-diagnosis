#!/bin/bash
set -e

python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Môi trường đã sẵn sàng. Kích hoạt bằng: source venv/bin/activate"
