import subprocess

experiments = [
    "exp1_shutdown_noise.py",
    "exp2_off_switch_game.py", 
    "exp3_linear_stress_test.py",
    "exp4_singularity_stress_test.py",
    "exp5_lambda_sweep.py",
    "exp6_reward_scaling_heatmap.py",
    "exp7_sequential_game.py"
]

for exp in experiments:
    print(f"\n{'='*50}")
    print(f"Running {exp}")
    print('='*50)
    subprocess.run(["python", f"experiments/{exp}"])
