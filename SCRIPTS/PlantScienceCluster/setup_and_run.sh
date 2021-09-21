#########
# fenics docker is already set up on the plant science cluster only on node16
# login to Carlos account
#########
ssh cal72@hydrogen.plantsci.cam.ac.uk
ssh node16 # only node16 has this docker setup

#########
# check what container is available
#########
docker container ls
#CONTAINER ID   IMAGE                          COMMAND                  CREATED         STATUS      PORTS     NAMES
#e4bfd3f92801   quay.io/fenicsproject/stable   "/sbin/my_init --qui…"   17 months ago   Up 5 days             hibiscus

#########
# load fenics docker
#########
docker start hibiscus
# hibiscus
docker exec -t -i hibiscus /bin/bash
# root@e4bfd3f92801:/home/fenics#

######
# run the job
######

# !!!!don't mess up in any other folders!
cd /home/cal72/3LQI/

# Way 1: run with 10 cores
mpirun -n 10 python3 bbz.py
# Way 2 (recommended): run non-interactively
nice nohup ./fenics "mpirun -n 10 python3 bbz.py" &
# check result
tail nohup.out

#######
# set up fenics with conda
#######
conda create -n fenicsproject -c conda-forge fenics
source activate fenicsproject
