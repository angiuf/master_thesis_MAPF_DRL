cd /home/andrea/CODE/master_thesis_MAPF_DRL/baselines

# PRIMAL
echo "------------------------PRIMAL-----------------------"
cd ./PRIMAL
source /home/andrea/.virtualenvs/primal/bin/activate
python test_simple_warehouse_env.py
deactivate

# AB-MAPPER
echo "----------------------AB-MAPPER----------------------"
cd ../AB_Mapper/AB_Mapper
if [ -d ".venv" ]; then
    source .venv/bin/activate
    python test_ab_mapper_warehouse_env.py
    deactivate
else
    echo "Warning: .venv not found in AB_Mapper/AB_Mapper directory"
fi

# DCC
echo "-------------------------DCC-------------------------"
cd ../../DCC
if [ -d ".venv" ]; then
    source .venv/bin/activate
    python test_custom_env.py
    deactivate
else
    echo "Warning: .venv not found in DCC directory"
fi

# SCRIMP
echo "-----------------------SCRIMP-----------------------"
cd ../SCRIMP
if [ -d ".venv" ]; then
    source .venv/bin/activate
    python eval_custom_warehouse_env.py
    deactivate
else
    echo "Warning: .venv not found in SCRIMP directory"
fi

# CBSH2-RTC
echo "---------------------CBSH2-RTC----------------------"
cd ../CBSH2-RTC
if [ -d ".venv" ]; then
    source .venv/bin/activate
    python test_custom_env.py
    deactivate
else
    echo "Warning: .venv not found in CBSH2-RTC directory"
fi

# EECBS
echo "-----------------------EECBS------------------------"
cd ../EECBS
if [ -d ".venv" ]; then
    source .venv/bin/activate
    python test_custom_env.py
    deactivate
else
    echo "Warning: .venv not found in EECBS directory"
fi