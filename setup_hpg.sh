export PYTHONPATH=${PYTHONPATH}:${PWD}/
export BASE_PATH=${PWD}

if [[ "$HOSTNAME" == login*ufhpc ]]; then
    echo "Loading modules"
    module load python/3.8
    module load git
fi
