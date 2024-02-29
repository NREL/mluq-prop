#!/bin/bash
INPUTDECK=inputs/dimreduce3.in
REFMODELPATH=/projects/mluq/mluq-prop/models/experiments/splitBNN3/splitBNN3
JOBFILENAME=dimreducejobs_3layer.txt
NLAYERS=3
JOBSCRIT=dimreduce3.job

# Generate the job file.
echo "Generating job file for TACC Launcher..."
python write_dimreduce_jobfile.py --input ${INPUTDECK} --filename ${JOBFILENAME} --num_layers ${NLAYERS} --refmodel_path ${REFMODELPATH}

# Update the input deck with the appropriate architecture.
#WARNING! Manual step.

# Update the SLURM job file with the appropriate input deck and configuration.
echo "Please submit the job script to the queue."
echo "Job files are stored in the ./jobs directory. Available job files are..."
ls ./jobs
# cd ./jobs
# sbatch ${JOBSCRIPT}
