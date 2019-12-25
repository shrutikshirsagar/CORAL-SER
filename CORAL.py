import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import argparse
import scipy.linalg as sla
import pandas as pd
import glob
from importlib import reload
import sys







if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='CORAL domain adaptation')
	parser.add_argument('--source-data-path', type=str, default='./data/source/', metavar='Path', help='Path to source domain data')
	parser.add_argument('--target-data-path', type=str, default='./data/target/', metavar='Path', help='Path to target domain data')
	parser.add_argument('--out-path', type=str, default='./out/', metavar='Path', help='Path to output')
	parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
	parser.add_argument('--n-target-files', type=int, default=18, metavar='S', help='random seed (default: 1)')
	args = parser.parse_args()

	source_domain_files = glob.glob(args.source_data_path+'*.csv')
	target_domain_files = glob.glob(args.target_data_path+'*.csv')

	#########################################################
	# Read target data

	DT = None

	for i in range(min(args.n_target_files, len(target_domain_files))):
		data = pd.read_csv(target_domain_files[i], sep=',').values[:,2:].astype(float)
		if DT is not None:
			DT = np.concatenate([DT, data], 0)
		else:
			DT = data


	#########################################################
	# Compute target domain statistics

	CT = np.cov(DT, rowvar=False) + np.eye(DT.shape[1])

	#########################################################
	# Iterate over source domain files and perform adaptation

	for source_file in source_domain_files:

		DS = pd.read_csv(source_file, sep=',').values

		names, times, DS = DS[:,0:1], DS[:,1:2], DS[:,2:].astype(float)

		## Whitening

		CS = np.cov(DS, rowvar=False) + np.eye(DS.shape[1])
		DS = DS.dot( np.linalg.inv( sla.sqrtm(CS) ) )

		## Coloring

		DS_t = DS.dot( sla.sqrtm( CT ) )

		df=pd.DataFrame(np.concatenate([names, times, DS_t],1))
		df.to_csv(args.out_path+source_file.split('/')[-1].split('.')[0]+'.csv', index = False, header=None)
