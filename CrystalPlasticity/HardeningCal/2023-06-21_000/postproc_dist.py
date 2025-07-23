"""
Skript for Damask3 Postprocessing, including macro - micro strain and defgrad comparison for RVE with phase field Damage
For 1 - 3 Phases

Niklas Fehlemann, IMS
"""

import damask
import os
import glob
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import pyvista as pv
import yaml
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

result = damask.Result(r'grid_load.hdf5')
result.add_strain(F='F')
result.add_stress_Cauchy(F='F')
result.add_equivalent_Mises('sigma')
result.add_equivalent_Mises('epsilon_V^0.0(F)')

if not os.path.isdir('PostProc'):
	os.mkdir('PostProc')

path = os.getcwd() + r'/PostProc'


# Averaged Quantities
strain_eq = result.place('epsilon_V^0.0(F)')
stress_eq = result.place('sigma')
crss_eq = result.get('xi_sl')
e_stress_eq = result.place('P')

strain_vector = list()
stress_vector = list()
crss_vector = list()
e_stress_vector = list()

for _, strain in strain_eq.items():  
    strain_mean = np.abs(strain[:, 0, 0].mean())
    strain_vector.append(strain_mean) 
strain_vector = np.asarray(strain_vector) 

for _, stress in stress_eq.items():
    stress_mean = np.abs(stress[:, 0, 0].mean())
    stress_vector.append(stress_mean)  
stress_vector = np.asarray(stress_vector)
    
for key, value in crss_eq.items():
    crss_start = list()
    for subkey, subval in value.items():
        crss_start.append(np.mean(subval))
    crss_vector.append(np.mean(crss_start))

for _, stress in e_stress_eq.items():
    e_stress_mean = np.abs(stress[:, 0, 0].mean())
    e_stress_vector.append(e_stress_mean)  
e_stress_vector = np.asarray(e_stress_vector)

flowcurve = pd.read_csv(r'TrueStressStrainDP800.csv', header=None)
flowcurve = flowcurve[flowcurve[0] <= 0.15]
real_stress = flowcurve[1] * 1000000
real_strain = flowcurve[0]

# Interpolate exp-curve at x_vals_sim for RMSE
real_stress_interp = np.interp(x=strain_vector, xp=real_strain.to_numpy(), fp=real_stress.to_numpy())
print(real_stress_interp[2:]/ 1000000)
print(stress_vector[2:]/ 1000000)
print((real_stress_interp[2:] / 1000000) - (stress_vector[2:] / 1000000))

# Calc RMSE (if not finished, then till failure)
loss = mean_absolute_error(stress_vector[2:] / 1000000, real_stress_interp[2:]/ 1000000)
print(loss)

data = pd.DataFrame([strain_vector, stress_vector, e_stress_vector, crss_vector]).T
print(data)
data.to_csv(path + '/RVEStress.csv', index=False)

plt.style.use('default')
fig, ax = plt.subplots(1, 1, figsize=(8,6))

ax.set_title(f'Comp with DP800 Flowcurve - Loss: {loss: .2f}', fontsize=18)
ax.plot(strain_vector, stress_vector, linewidth=3, label='True Stress RVE')
ax.scatter(strain_vector[2:], stress_vector[2:])
ax.plot(real_strain.to_numpy(), real_stress.to_numpy(), label='DP800 Flowcurve', linewidth=3)
ax.scatter(strain_vector[2:], real_stress_interp[2:], marker='x', s=100)
ax.plot(strain_vector, e_stress_vector, linewidth=3, label='Eng. Stress RVE')
ax.set_xlabel('True Strain (-)', fontsize=16)
ax.set_ylabel('Stress (MPa)', fontsize=16)
ax.set_ylim([0, 14e8])
ax.set_xlim([-0.001, 0.20])
ax.tick_params(which='both', size=10, labelsize=16)
ax.legend(fontsize=18)

plt.tight_layout()
fig.savefig(path + r'/AveragedCurves.svg')


# CRSS Distribution
crss_dist_start= result.view('increment', 2).get('xi_sl')
crss_dist_96 = result.view('increment', 100).get('xi_sl')
crss_dist_146 = result.view('increment', 154).get('xi_sl')
#crss_dist_216 = result.view('increment', -1).get('xi_sl')

crss_start = list()
for key, val in crss_dist_start.items():
    crss_start.append(np.mean(val)*1e-6)

crss_96 = list()
for key, val in crss_dist_96.items():
    crss_96.append(np.mean(val)*1e-6)

crss_146 = list()
for key, val in crss_dist_146.items():
    crss_146.append(np.mean(val)*1e-6)

#crss_216 = list()
#for key, val in crss_dist_216.items():
#    crss_216.append(np.mean(val)*1e-6)

pd.DataFrame(crss_start).to_csv(path + '/crss_start.csv', index=False)
pd.DataFrame(crss_96).to_csv(path + '/crss_96.csv', index=False)
pd.DataFrame(crss_146).to_csv(path + '/crss_146.csv', index=False)
#pd.DataFrame(crss_216).to_csv(path + '/crss_216.csv', index=False)

crss_exp_0 = [254.748,216.7323,259.7,339.885,200.514,248.577,220.7112,208.65,309.0607,199.287,218.0766,235.296,193.7785,228.8589,252.2484,178.3394,274.618,193.248,185.6745,159.938,193.9392,190.1994]
crss_exp_96 = [331.7304,349.0482,284.553,275.3268,273.955,322.896,263.424,262.911,256.8606,294.1824,255.0196,349.92,304.608,275.95,282.5064]
crss_exp_146 = [308.6,262.2422,343.1715,299.5939,286.9646,330.3272,329.6416,321.44,297.3054,343.0186,318.3018,359.094,344.7171,399.4779,288.9396,350.3448,336.1708,290.7474]
#crss_exp_216 = [420.24,348.9424,348.1992,344.6198,301.0392,321.4099,358.9596,421.344,379.2,401.9202,491.9096,407.2881,441.2501,412.0358,373.8098,438.1355,470.7612,319.7544,332.8078,365.6581,364.041,422.7904]

strain_0 = strain_vector[1]
strain_96 = strain_vector[50]
strain_146 = strain_vector[77]
#strain_216 = strain_vector[-1]
fig, ax = plt.subplots(1,1, figsize=(8,6))
sns.ecdfplot(x=crss_exp_0, label='Exp CRSS at 0.0%', ax=ax, linewidth=6, linestyle=':', color='tab:blue')
sns.ecdfplot(x=crss_start, label=f'Sim CRSS at {strain_0:.2%}', ax=ax, linewidth=4, linestyle='-', color='tab:blue')

sns.ecdfplot(x=crss_exp_96, label='Exp CRSS at 9.6%', ax=ax, linewidth=6, linestyle=':', color='tab:orange')
sns.ecdfplot(x=crss_96, label=f'Sim CRSS at {strain_96:.2%}', ax=ax, linewidth=4, linestyle='-', color='tab:orange')

sns.ecdfplot(x=crss_exp_146, label='Exp CRSS at 14.6%', ax=ax, linewidth=6, linestyle=':', color='tab:green')
sns.ecdfplot(x=crss_146, label=f'Sim CRSS at {strain_146:.2%}', ax=ax, linewidth=4, linestyle='-', color='tab:green')

#sns.ecdfplot(x=crss_exp_216, label='Exp CRSS at 21.6%', ax=ax, linewidth=6, linestyle=':', color='tab:red')
#sns.ecdfplot(x=crss_216, label=f'Sim CRSS at {strain_216:.2%}', ax=ax, linewidth=4, linestyle='-', color='tab:red')
ax.legend(fontsize=14)
ax.set_xlabel('CRSS (MPa)', fontsize=16)
ax.set_ylabel('Proportion', fontsize=16)
ax.tick_params(which='both', size=10, labelsize=16)
ax.set_xlim([150, 550])

plt.tight_layout()
fig.savefig(path + '/CRSS_Comparison.svg')


# Plot Comparison
real_values = {0 : 226, 9.6*2.75/100 : 292, 14.6*2.75/100 : 323, 21.6*2.75/100 : 386}

sim_values = {0 : float(np.mean(crss_start)),
             strain_96*2.75 : float(np.mean(crss_96)),
             strain_146*2.75 : float(np.mean(crss_146))}

mean_hardening_15 = (np.mean(crss_146) - np.mean(crss_start)) / (strain_146*2.75 - 0)

fig, ax = plt.subplots(1, 1, figsize=(8,6))
ax.set_title(f'Hardening (15%): {mean_hardening_15:.2f}')
ax.plot(real_values.keys(), real_values.values(), label='Experimental')
ax.scatter(real_values.keys(), real_values.values(), marker='o', s=180)

ax.plot(sim_values.keys(), sim_values.values(), label='Simulative', linestyle='--')
ax.scatter(sim_values.keys(), sim_values.values(), marker='x', s=180)

ax.set_xlabel('Shear Strain', fontsize=16)
ax.set_ylabel('CRSS (MPA)', fontsize=16)
ax.set_ylim([200, 550])
ax.tick_params(which='both', size=10, labelsize=16)
ax.legend(fontsize=18)

plt.tight_layout()
fig.savefig(path + r'/StrainHardening_sim_exp.svg')


# Partitioned Flowcurves
ferrite_stress = list()
martensite_stress = list()
for key, val in result.get('sigma').items():
    temp = list()
    for subkey, subvalue in val.items():
        if 'Ferrite' in subkey:
            temp.append(np.mean(subvalue[:,0,0]))
        else:
            martensite_stress.append(np.mean(subvalue[:,0,0]))
    ferrite_stress.append(np.mean(temp))
    
ferrite_strain = list()
martensite_strain = list()
for key, val in result.get('epsilon_V^0.0(F)').items():
    temp = list()
    for subkey, subvalue in val.items():
        if 'Ferrite' in subkey:
            temp.append(np.mean(subvalue[:,0,0]))
        else:
            martensite_strain.append(np.mean(subvalue[:,0,0]))
    ferrite_strain.append(np.mean(temp))

fig, ax = plt.subplots(1, 1, figsize=(8,6))
    
ax.plot(ferrite_strain, ferrite_stress, label='Avg. Stress Ferrite', linewidth=4)
ax.plot(martensite_strain, martensite_stress, label='Avg. Stress Martensite', linewidth=4)
ax.legend(fontsize=18)
ax.set_xlabel('Strain', fontsize=16)
ax.set_ylabel('Stress', fontsize=16)
ax.tick_params(which='both', size=10, labelsize=16)

plt.tight_layout()
fig.savefig(path + r'/FlowcurvePartitioning.svg')

"""
# Calculate WassDistance
diff_0 = stats.wasserstein_distance(crss_start, crss_exp_0)
diff_96 = stats.wasserstein_distance(crss_96, crss_exp_96)
diff_146 = stats.wasserstein_distance(crss_146, crss_exp_146)
diff_216 = stats.wasserstein_distance(crss_216, crss_exp_216)
print('++++++++++++++++++++++++++++++++++')
print(diff_0, diff_96, diff_146, diff_216)
print('++++++++++++++++++++++++++++++++++')
combined = (diff_0 + diff_96 + diff_146 + diff_216) / 4

# Store Values
final_strain = strain_vector[-1]
with open(path + r'/report.txt', 'w') as report:
    report.writelines('Final Distributions: \n\n')

    report.writelines(f'CRSS-mean at 0% Strain: {float(np.mean(crss_start)):.4} MPa \n')
    report.writelines(f'CRSS-std at 0% Strain: {float(np.std(crss_start)):.4} MPa \n\n')

    report.writelines(f'CRSS-mean at {strain_96:.2%} Strain: {float(np.mean(crss_96)):.4} MPa \n')
    report.writelines(f'CRSS-std at {strain_96:.2%} Strain: {float(np.std(crss_96)):.4} MPa \n\n')

    report.writelines(f'CRSS-mean at {strain_146:.2%} Strain: {float(np.mean(crss_146)):.4} MPa \n')
    report.writelines(f'CRSS-std at {strain_146:.2%} Strain: {float(np.std(crss_146)):.4} MPa \n\n')

    report.writelines(f'CRSS-mean at {strain_216:.2%} Strain: {float(np.mean(crss_216)):.4} MPa \n')
    report.writelines(f'CRSS-std at {strain_216:.2%} Strain: {float(np.std(crss_216)):.4} MPa \n\n')

    report.writelines(f'Wasserstein-losses: {diff_0:.2f}/{diff_96:.2f}/{diff_146:.2f}/{diff_216:.2f} \n')
    report.writelines(f'Wasserstein-loss combined: {combined:.2f}')
"""








