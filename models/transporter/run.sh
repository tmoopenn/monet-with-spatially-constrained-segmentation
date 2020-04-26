
python3 -m venv transporter-venv
source transporter-venv/bin/activate
pip install -r transporter/requirements.txt
python -m transporter.transporter_test
