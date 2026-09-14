#!/bin/bash

python callibrate_K_bash.py --lhs_n 250000 --auger_id DB2 &
python callibrate_K_bash.py --lhs_n 250000 --auger_id DB4 &
python callibrate_K_bash.py --lhs_n 250000 --auger_id DB6 &
python callibrate_K_bash.py --lhs_n 250000 --auger_id DL3 &
python callibrate_K_bash.py --lhs_n 250000 --auger_id DL4 &
python callibrate_K_bash.py --lhs_n 250000 --auger_id DL5 &
python callibrate_K_bash.py --lhs_n 250000 --auger_id DBSP2 &
python callibrate_K_bash.py --lhs_n 250000 --auger_id DBSP3 &
python callibrate_K_bash.py --lhs_n 250000 --auger_id DBSP4 &
wait
