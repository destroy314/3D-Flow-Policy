# bash scripts/gen_demonstration_metaworld.sh basketball
# bash scripts/gen_demonstration_metaworld.sh assembly
# bash scripts/gen_demonstration_metaworld.sh shelf-place

# assembly basketball bin-picking box-close button-press-topdown button-press-topdown-wall button-press button-press-wall coffee-button coffee-pull coffee-push dial-turn disassemble door-close door-lock door-open door-unlock hand-insert drawer-close drawer-open faucet-open faucet-close hammer handle-press-side handle-press handle-pull-side handle-pull lever-pull peg-insert-side pick-place-wall pick-out-of-hole reach push-back push pick-place plate-slide plate-slide-side plate-slide-back plate-slide-back-side peg-unplug-side soccer stick-push stick-pull push-wall reach-wall shelf-place sweep-into sweep window-open window-close

cd third_party/Metaworld

task_name=${1}

export CUDA_VISIBLE_DEVICES=0
python gen_demonstration_expert.py --env_name=${task_name} \
            --num_episodes 10 \
            --root_dir "../../3D-Diffusion-Policy/data/" \
