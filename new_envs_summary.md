# Phase 4 Run Summary

## baseline_dynobs

- runs: `10`
- mean reward all episodes: `-0.1893 +/- 0.3746`
- mean reward last 20 episodes: `0.0423 +/- 0.3004`
- positive episodes: `18369 / 31436`
- best observed training reward: `0.975`

## flowreg_dynobs

- runs: `10`
- mean reward all episodes: `-0.1848 +/- 0.3774`
- mean reward last 20 episodes: `0.0771 +/- 0.3149`
- positive episodes: `22181 / 35113`
- best observed training reward: `0.975`
- FlowReg scalars finite: `True`
- FlowReg updates: `480` logged, `480` expected
- Loss/FlowReg: `14.9922 +/- 4.9305`
- Loss/FlowReg_PaperScaled: `14.9922 +/- 4.9305`
- Loss/FlowReg_MSEMean: `0.2343 +/- 0.0770`
- Latent/Path_Length: `3.5431 +/- 0.6539`
- Latent/Net_Displacement: `0.7095 +/- 0.2826`
- Latent/Acceleration_Energy: `6.3162 +/- 1.2717`
- Latent/ODE_Error_Drift: `4.7696 +/- 0.9395`

## baseline_doorkey

- runs: `10`
- mean reward all episodes: `0.0037 +/- 0.0097`
- mean reward last 20 episodes: `0.0044 +/- 0.0140`
- positive episodes: `27 / 3894`
- best observed training reward: `0.981`

## flowreg_doorkey

- runs: `10`
- mean reward all episodes: `0.0029 +/- 0.0053`
- mean reward last 20 episodes: `0.0020 +/- 0.0063`
- positive episodes: `20 / 3891`
- best observed training reward: `0.922`
- FlowReg scalars finite: `True`
- FlowReg updates: `480` logged, `480` expected
- Loss/FlowReg: `6.3995 +/- 5.9075`
- Loss/FlowReg_PaperScaled: `6.3995 +/- 5.9075`
- Loss/FlowReg_MSEMean: `0.1000 +/- 0.0923`
- Latent/Path_Length: `0.7538 +/- 1.7905`
- Latent/Net_Displacement: `0.0989 +/- 0.1764`
- Latent/Acceleration_Energy: `1.4684 +/- 3.5525`
- Latent/ODE_Error_Drift: `3.3128 +/- 1.1454`
