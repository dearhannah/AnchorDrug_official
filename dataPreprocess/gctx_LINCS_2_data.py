"""
Modified from Jing Xing's paper
"""
import os
import pandas as pd
import numpy as np
import h5py


def _gctx2dataframe(dir_input, file_name_gctx, geneinfo_beta):
    """
    Transfer gctx file to csv (dataframe) and pick landmark genes.
    _input:
    gctx_file_name: str, the gctx file name
    dir_input: str, the input directory where the gctx file locates.
    geneinfo_beta: str, file name, the file contains detailed gene information such as whether a gene is landmark gene.
    _output:
    df: pd.DataFrame, a dataframe that contains z-score (data) for landmark gene id (columns) and signature id (index)
    """
    # Load Z-score matrix file
    data_ori = h5py.File(dir_input + file_name_gctx, 'r')
    row_id = data_ori['0']['META']['ROW']['id'][:].astype(int)
    col_id = data_ori['0']['META']['COL']['id'][:].astype(str)
    data_matrix = data_ori['0']['DATA']['0']['matrix']
    print('Number of rowid:', len(row_id))
    print('Number of colid:', len(col_id))
    print('Z-scores loaded:', data_matrix.shape)
    # Load 978 landmark gene IDs
    gene_info_ori = pd.read_csv(dir_input + geneinfo_beta, sep='\t', index_col='gene_id')
    gene_info = gene_info_ori[gene_info_ori.feature_space == 'landmark']
    print('Number of landmark genes:', gene_info.shape[0])
    # Keep landmark genes
    row_dict = dict([(i, gid) for i, gid in enumerate(row_id) if gid in gene_info.index])
    landmark_index = list(row_dict.keys())
    row_id_landmark = list(row_dict.values())
    gene_id_landmark = gene_info['gene_symbol'][row_id_landmark]
    data_matrix_landmark = data_matrix[:, landmark_index]
    # output csv file
    df = pd.DataFrame(data=data_matrix_landmark, columns=gene_id_landmark, index=col_id)
    print('shape of data matrix', df.shape)
    return df


def _pick_signature_time_dose(Time, Dose, df_z_data, siginfo_beta):
    """
    Pick signatures needed (time and dose) from dataframe of z-score.
    _input:
    Time, Dose: str, time and dose needed
    df_z_data: pd.DataFrame, a dataframe of z-score (data) for landmark gene id (columns) and signature id (index)
    siginfo_beta: str, filename of a signature information (such as quality, dose and time) file
    _output:
    df_z_data_gold_time_dose: pd.DataFrame, a dataframe of z-score (data) for landmark gene id (columns)
                             and quality=gold & selected-time-dose signature id (index)
    """
    print("time: %s and dose: %s" % (Time, Dose))
    # Load signature meta table
    sig_info_ori = pd.read_csv(siginfo_beta, sep='\t', index_col='sig_id', low_memory=False)
    sig_info = sig_info_ori.loc[[idx for idx in df_z_data.index]]
    # Keep "gold" signatures
    sig_info_gold = sig_info[(sig_info['cc_q75'] >= 0.2) & (sig_info['pct_self_rank_q25'] <= 5)]
    # sig_info_gold = sig_info
    # Select needed time
    if Time == '24h':
        sig_info_gold_time = sig_info_gold[(sig_info_gold['pert_time_unit'] == 'h')
                                           & (sig_info_gold['pert_time'] == 24)]
    # Select needed dose
    if Dose == '10uM':
        sig_info_gold_time_dose = sig_info_gold_time[(sig_info_gold_time['pert_dose_unit'] == 'uM')
                                                     & (sig_info_gold_time['pert_dose'] == 10)]
    elif Dose == '0.5-2uM':
        sig_info_gold_time_dose = sig_info_gold_time[(sig_info_gold_time['pert_dose_unit'] == 'uM')
                                                     & (sig_info_gold_time['pert_dose'] <= 2)
                                                     & (sig_info_gold_time['pert_dose'] >= 0.5)]
    elif Dose == '0.01-2uM':
        sig_info_gold_time_dose = sig_info_gold_time[(sig_info_gold_time['pert_dose_unit'] == 'uM')
                                                     & (sig_info_gold_time['pert_dose'] <= 2)
                                                     & (sig_info_gold_time['pert_dose'] >= 0.01)]
    else:
        raise Exception('Unknown dose!')
    # Check output
    if len(sig_info) == 0:
        raise Exception('Nothing left after filtering!')
    # prepare output dataframe
    df_z_data_gold_time_dose = df_z_data.loc[[idx for idx in sig_info_gold_time_dose.index]]
    print('All z-score can be mapped to signature info:',
          len(df_z_data_gold_time_dose.index) == sum(df_z_data_gold_time_dose.index == sig_info_gold_time_dose.index))
    return df_z_data_gold_time_dose, sig_info_gold_time_dose


def _add_cell_drug2z(df_z, df_siginfo, compound_info_check):
    """
    Add compound and cell line information to z-score matrix
    """
    # Load compound information
    cpd_info_ori = pd.read_csv(compound_info_check, index_col='pert_id')
    cpd_info = cpd_info_ori[~cpd_info_ori.index.duplicated()]
    print('Number of compounds:', cpd_info.shape[0])
    # Tailor siginfo using compound information
    df_siginfo_sub = df_siginfo[['pert_id', 'cell_iname']]
    df_siginfo_sub = df_siginfo_sub[df_siginfo_sub['pert_id'].isin(cpd_info.index)]
    # Add compound info to siginfo
    cpd_info_2sig = cpd_info.loc[df_siginfo_sub['pert_id']]
    print('All compounds can be mapped to signature info: ',
          len(df_siginfo_sub['pert_id']) == sum(cpd_info_2sig.index == df_siginfo_sub['pert_id']))
    df_cd = df_siginfo_sub
    df_cd['SMILES'] = cpd_info_2sig['SMILES'].to_list()
    # Tailor z-score
    df_z = df_z.loc[df_cd.index]
    print('All z-score can be mapped to signature info:', len(df_cd.index) == sum(df_cd.index == df_z.index))
    df_cdz = pd.concat([df_cd, df_z], axis=1)
    print("Size of output dataframe: ", df_cdz.shape)
    return df_cdz


def gctx2PRE_ORI_DATA():
    """

    """
    # preparation
    dir_input = '/Users/menghan1/pretrain_paper/data/LINCS2020_l5_cmpd/'
    file_name_gctx = 'level5_beta_trt_cp_n720216x12328.gctx'
    # change time and dose as need
    Time, Dose = '24h', '10uM'
    # Time, Dose = '24h', '0.5-2uM'
    dir_output = '/Users/menghan1/pretrain_paper/data/LINCS2020_l5_cmpd_%s_%s/' % (Time, Dose)
    if not os.path.exists(dir_output):
        os.mkdir(dir_output)
    file_name_outdf = 'level5_beta_trt_cp_%s_%s.csv' % (Time, Dose)
    geneinfo_beta = 'geneinfo_beta.txt'
    compound_info_check = 'compoundinfo_checked.csv'
    siginfo_beta = 'siginfo_beta.txt'
    print("Preparation done!")
    print("Transferring gctx to dataframe...")
    # Transfer gctx file to csv (dataframe) and select landmark genes
    df_data = _gctx2dataframe(dir_input, file_name_gctx, geneinfo_beta)
    # df_data.to_csv(dir_input + 'level5_beta_trt_cp.csv')
    print("Picking needed signatures...")
    # Pick signatures needed
    df_data_pick, df_siginfo_pick = _pick_signature_time_dose(Time, Dose, df_data, dir_input + siginfo_beta)
    print("Mapping compound to signature and z-score...")
    # Add compound (smiles, drug id) and cell line (cell id) information to z-score matrix
    df_cdz = _add_cell_drug2z(df_data_pick, df_siginfo_pick, dir_input + compound_info_check)
    print("Finishing")
    # output
    df_cdz.to_csv(dir_output + file_name_outdf)


if __name__ == '__main__':
    gctx2PRE_ORI_DATA()
