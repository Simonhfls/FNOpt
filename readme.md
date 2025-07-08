## Meta-FNO

The trained Meta-FNO model is available for download at:
https://we.tl/t-ADSmTRQVzV


After downloading the model, you can put it in the Logger/ folder. 


To run SfT code:

Local: Run `main_sft.py` directly.
Note: for local run, I only use 3 frames just for debugging, otherwise loading ground truth is a bit slow.

On Cluster:
Run main_sft.py with the following command:

```bash
ppython main_sft.py --config_file=fno_vertex.yaml --config_name=e14 \
		--inference.sft.debug 0 \
		--inference.visualize_scaling 0 \
		--inference.visualize_grads 0 \
		--inference.sft.evaluate 1 \
		--inference.material.stretching 1000 \
		--inference.sft.lr.stretching 0.5 \
		--inference.sft.lr.shearing 0.1 \
		--inference.sft.lr.bending 0.005 \
		--inference.sft.lr.external 0.05 \
		--inference.sft.lr.vertex 0.01 \
		--inference.sft.lc.rgb 1 \
		--inference.sft.lc.sil 1 \
		--inference.sft.lc.shift 0.01 \
		--inference.sft.optimize_uv 0 \
		--inference.sft.iterations_per_timestep 20 \
		--inference.sft.new_frame_period 5 \
		--inference.sft.n_epochs_opt 300
``` 

The entire slurm file (fno_sft.sh) that can be launch by `sbatch fno_sft.sh`
```bash
#!/bin/bash
#SBATCH --job-name=04b_sft_fno
#SBATCH -A dnt@v100
##SBATCH -A dnt@a100
##SBATCH -C a100
##SBATCH -C v100-32g
#SBATCH --partition=gpu_p2,gpu_p13
#SBATCH --ntasks=1 		     # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1			# number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH -t 2:00:00
#SBATCH --qos=qos_gpu-dev
#SBATCH --output=/lustre/fswork/projects/rech/dnt/uok26jj/liris_code/slurm/metamizer/jobmainsft.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ruochen.chen@ec-lyon.fr

echo "Node: $(hostname)"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"

module purge
module load pytorch-gpu/py3/2.3.0

set -x


cd /lustre/fswork/projects/rech/dnt/uok26jj/liris_code/Metamizer

# e14, e11, or e16?

python main_sft.py --config_file=fno_vertex.yaml --config_name=e14 \
		--inference.sft.debug 0 \
		--inference.visualize_scaling 0 \
		--inference.visualize_grads 0 \
		--inference.sft.evaluate 1 \
		--inference.material.stretching 1000 \
		--inference.sft.lr.stretching 0.5 \
		--inference.sft.lr.shearing 0.1 \
		--inference.sft.lr.bending 0.005 \
		--inference.sft.lr.external 0.05 \
		--inference.sft.lr.vertex 0.01 \
		--inference.sft.lc.rgb 1 \
		--inference.sft.lc.sil 1 \
		--inference.sft.lc.shift 0.01 \
		--inference.sft.optimize_uv 0 \
		--inference.sft.iterations_per_timestep 20 \
		--inference.sft.new_frame_period 5 \
		--inference.sft.n_epochs_opt 300 \

```
