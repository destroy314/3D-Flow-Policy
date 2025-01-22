from pynvml import *
import subprocess
import time, os
import signal

max_instance_per_device = 1
max_instance_sametime = 4
banned_devices = [0,1]
vram_needed_GB = 10
timeout_minute = float('inf')

# fill command and name of tasks here
tasks = []
task_names = []
task_env = {}
task_cwd = None

task_env["HYDRA_FULL_ERROR"] = "1"
task_env["TQDM_DISABLE"] = "1"
task_cwd = "/data22/DISCOVER_summer2023/yangz2306/3D-Diffusion-Policy/3D-Diffusion-Policy"

def make_command(policy, env, seed=0, future=False):
    return f"python train.py \
            --config-name={policy}.yaml \
            task={env} \
            hydra.run.dir=data/outputs/{env}-{policy}_seed{seed} \
            training.debug=False \
            training.seed={seed} \
            training.device='cuda:0' \
            exp_name={env}-{policy} \
            logging.mode=offline \
            checkpoint.save_ckpt=False \
            +policy.pred_future={future}"

seeds = [0]
# policies=["simple_dp3_dit_future", "simple_dp3_future"]
# envs=["adroit_door", "adroit_pen", "metaworld_disassemble", "metaworld_stick-pull", "metaworld_pick-place-wall"]
policies=["dp3_dit", "dp3"]
envs=["adroit_door", "metaworld_stick-pull"]

for seed in seeds:
    for policy in policies:
        for env in envs:
            tasks.append(make_command(policy, env, seed))
            task_names.append(f"{env}_{policy}_{seed}")

            tasks.append(make_command(policy, env, seed, True))
            task_names.append(f"{env}_{policy}_future_{seed}")


nvmlInit()
deviceCount = nvmlDeviceGetCount()

instance_count = [0] * deviceCount
trainers = []


def signal_handler(signal, frame):
    for t in trainers:
        os.killpg(os.getpgid(t.p.pid), 9)
        print(f"kill {t.name}")
    for t in trainers:
        t.fp.close()
        t.p.wait()
        print(f"{t.name} exited")
    sys.exit(1)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGHUP, signal_handler)


class Train:
    def __init__(self, args, name, device):
        print(f"running {name} on GPU {device}")
        self.name = name
        self.fp = open(f"runlogs/{int(time.time())}_{name}.log", "w")
        self.device = device
        instance_count[device] += 1
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = f"{device}"
        my_env.update(task_env)
        self.p = subprocess.Popen(
            args,
            stdout=self.fp,
            stderr=self.fp,
            env=my_env,
            cwd=task_cwd,
            preexec_fn=os.setsid,
        )
        self.start_time = time.time()

    def is_end(self):
        duration_minute = int((time.time() - self.start_time) / 60)
        if self.p.poll() is not None:
            print(
                f"{self.name} exited with code {self.p.poll()} after {duration_minute} minutes."
            )
            self.fp.close()
            instance_count[self.device] -= 1
            return True
        if duration_minute > timeout_minute:
            print(f"timeout kill {self.name}")
            self.p.terminate()
            self.fp.close()
            instance_count[self.device] -= 1
            return True
        return False


os.makedirs("runlogs", exist_ok=True)
start_time = time.time()
while tasks or trainers:
    for trainer in trainers:
        if trainer.is_end():
            trainers.remove(trainer)
    if len(trainers) >= max_instance_sametime or not tasks:
        time.sleep(60)
        continue
    task = tasks.pop()
    name = task_names.pop()
    exec = False
    for i in range(deviceCount):
        if i in banned_devices:
            continue
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        if (int(info.free) / 1024**3) - instance_count[
            i
        ] * vram_needed_GB > vram_needed_GB and instance_count[
            i
        ] < max_instance_per_device:
            args = task.split()
            trainer = Train(args, name, i)
            trainers.append(trainer)
            exec = True
            break
    if not exec:
        tasks.append(task)
        task_names.append(name)
        time.sleep(60)
print(
    f"all tasks finished after {int((time.time() - start_time) / 60)} minutes."
)

nvmlShutdown()