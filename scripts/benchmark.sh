# sleep 4h
# policies=(simple_dp3_sit simple_dp3_dit simple_dp3)
policies=(simple_dp3)
time=$(date "+%m%d-%H%M")
# tasks=(adroit_hammer adroit_door adroit_pen metaworld_assembly metaworld_basketball metaworld_shelf-place)
# tasks=(metaworld_assembly metaworld_basketball metaworld_shelf-place)
# tasks=(adriot_door adroit_pen metaworld_shelf-place metaworld_disassemble metaworld_stick-pull metaworld_pick-place-wall) # hard?
tasks=(metaworld_disassemble metaworld_stick-pull metaworld_pick-place-wall) # hard?
for policy in "${policies[@]}"; do
    for task in "${tasks[@]}"; do
        bash scripts/train_policy.sh $policy $task $time 0
    done
done

policies=(simple_dp3_dit_future simple_dp3_future)
tasks=(adriot_door adroit_pen metaworld_shelf-place metaworld_disassemble metaworld_stick-pull metaworld_pick-place-wall) # hard?
for policy in "${policies[@]}"; do
    for task in "${tasks[@]}"; do
        bash scripts/train_policy.sh $policy $task $time 0
    done
done