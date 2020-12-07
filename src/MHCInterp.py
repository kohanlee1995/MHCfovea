import os, sys, re, json, random
import numpy as np
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import logomaker as lm
from sklearn.manifold import TSNE
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram
import warnings
warnings.filterwarnings('ignore')


class MHCInterp():
    def __init__(self, mhc_seq_dict, mhc_motif_dict, submotif_len, position_dict, output_dir):
        self.aa_str = "ACDEFGHIKLMNPQRSTVWY"
        self.mhc_seq_dict = mhc_seq_dict
        self.mhc_motif_dict = mhc_motif_dict
        self.submotif_len = submotif_len
        self.position_dict = position_dict
        self.output_dir = output_dir
        
        pos = self.position_dict['selected']
        for hla in ['A', 'B', 'C']:
            pos = list(set(pos) & set(self.position_dict[hla]))
        self.position_dict['intersection'] = sorted(pos)

        self.dpi = 600
        self.fontsize = 10
        
        self.Nside_df = None
        self.Cside_df = None


    def Clustering(self, hla, side, noise_threshold,
                   clustering_method, clustering_kwargs,
                   reduction_method=None, reduction_kwargs=None,
                   pre_pdist=False, metric='cosine', method='average',
                   highlight=False, load_file=False, turn_off_label=False):
        
        alleles = sorted([i for i in list(self.mhc_motif_dict.keys()) if hla in i])

        # motif_df
        if load_file:
            motif_df = pd.read_csv('%s/%s_%ssideDF.csv'%(self.output_dir, hla, side), index_col=0)
        else:
            motif_df = self._submotif_df(alleles, side, noise_threshold)
            # pdist
            if pre_pdist:
                motif_df = self._submotif_pdist(motif_df, metric)
            # feature reduction
            if reduction_method:
                _, motif_df = self._submotif_feature_reduction(motif_df, reduction_method, reduction_kwargs)

        # clustering
        _, motif_df = self._submotif_clustering(motif_df, clustering_method, clustering_kwargs)

        # clustermap plot
        g = self._clustermap_plot(motif_df, metric, method, figfile='%s/%s_%ssidePDIST.png'%(self.output_dir, hla, side))
        
        # reset group order
        group_list = list(motif_df.iloc[g.dendrogram_row.reordered_ind]['label'])
        group_list = sorted(set(group_list), key=group_list.index)
        current_group = 0
        for i in range(len(group_list)):
            if i == -1:
                motif_df['label'].replace(group_list[i], 'group_-1')
            else:
                motif_df['label'].replace(group_list[i], 'group_%d'%current_group, inplace=True)
                current_group+=1
        motif_df.to_csv('%s/%s_%ssideDF.csv'%(self.output_dir, hla, side))
        print(motif_df['label'].value_counts())
        
        # main plot
        self._main_plot(hla, side, motif_df, ylim=4, highlight=highlight, turn_off_label=turn_off_label,
                        figfile='%s/%s_%ssideMain.png'%(self.output_dir, hla, side))
        
        # embedding plot
        if reduction_method:
            self._emb_plot(motif_df, hue='label', figfile='%s/%s_%ssideEMB.png'%(self.output_dir, hla, side))
        
        print("%s-side of HLA-%s complete"%(side, hla))
    
    
    def AlleleGrouping(self, hla):
        Nside_df = pd.read_csv('%s/%s_NsideDF.csv'%(self.output_dir, hla), index_col=0)
        Cside_df = pd.read_csv('%s/%s_CsideDF.csv'%(self.output_dir, hla), index_col=0)
        
        # group count
        Nside_group_df = self._group_count(Nside_df)
        Cside_group_df = self._group_count(Cside_df)
        df_dict = {'Nside': Nside_group_df, 'Cside': Cside_group_df}
        
        for key, df in df_dict.items():
            print(key)
            for idx, row in df.iterrows():
                groups = row[row >= 25].index
                groups = ['%s%s'%(j[0], j[2:]) for j in groups]
                groups_str = ' '.join(groups)
                print(idx, groups_str)
        
        # plot
        fig, ax = plt.subplots(2,1, figsize=(8, 4), dpi=self.dpi)
        self._group_count_plot(Nside_group_df, ax[0], xticklabel=False)
        self._group_count_plot(Cside_group_df, ax[1])
        fig.tight_layout()
        fig.savefig('%s/%s_GroupCount.png'%(self.output_dir, hla))
        
        return Nside_group_df, Cside_group_df
        
        
    def Analysis(self, hla, Nside_group=None, Cside_group=None, side='both',
                 highlight_pos_dict=dict(), turn_off_label=False):
        Nside_df = pd.read_csv('%s/%s_NsideDF.csv'%(self.output_dir, hla), index_col=0)
        Cside_df = pd.read_csv('%s/%s_CsideDF.csv'%(self.output_dir, hla), index_col=0)
        figfile = '%s/%s_'%(self.output_dir, hla)
        
        # alleles
        if Nside_group != None:
            Nside_alleles = Nside_df[Nside_df['label']=='group_%d'%Nside_group].index
            figfile += 'N%d'%Nside_group
        else:
            Nside_alleles = Nside_df.index
        
        if Cside_group != None:
            Cside_alleles = Cside_df[Cside_df['label']=='group_%d'%Cside_group].index
            figfile += 'C%d'%Cside_group
        else:
            Cside_alleles = Cside_df.index
        
        alleles = list(set(Nside_alleles) & set(Cside_alleles))
        print("Allele Number: ", len(alleles))
        
        # plot
        if side == 'both':
            fig, ax = plt.subplots(1, 2, figsize=(7.5, 1.5), dpi=self.dpi, gridspec_kw={'width_ratios': [2, 3]})
        else:
            fig, ax = plt.subplots(1, 2, figsize=(6, 1.5), dpi=self.dpi, gridspec_kw={'width_ratios': [1, 3]})
        hla_seqlogo_df = self._mhc_seqlogo(Nside_df.index, self.position_dict['selected'])
        self._motif_plot(alleles, side, ax[0], turn_off_label=turn_off_label)
        self._mhcseq_plot(alleles, self.position_dict['selected'], ax[1], diff_df=hla_seqlogo_df,
                          highlight_pos_dict=highlight_pos_dict, turn_off_label=turn_off_label)

        fig.tight_layout()
        fig.savefig(figfile)


    """"""""""""""""""""""""""""""""""""""
    # Minor Functions
    """"""""""""""""""""""""""""""""""""""
    def _submotif_df(self, alleles, side, noise_threshold):
        motif_df = pd.DataFrame(columns=alleles)
        for allele in alleles:
            # set noise to 0
            motif_arr = self.mhc_motif_dict[allele].copy()
            motif_arr[motif_arr < noise_threshold] = 0
            # side
            if side == 'N':
                arr = motif_arr[:self.submotif_len].flatten()
            elif side == 'C':
                arr = motif_arr[-self.submotif_len:].flatten()
            else:
                print("wrong side input")
                assert ValueError
            # motif_df
            motif_df[allele] = pd.Series(arr)
        motif_df = motif_df.T
        motif_df = motif_df.loc[:, (motif_df != 0).any(axis=0)]
        return motif_df


    def _submotif_feature_reduction(self, motif_df, method, kwargs):
        if method == 'UMAP':
            emb = UMAP(n_neighbors=kwargs['UMAP_n_neighbors'], min_dist=kwargs['UMAP_min_dist'],
                       n_components=2, random_state=0)
        elif method == 'tSNE':
            emb = TSNE(perplexity=kwargs['TSNE_perplexity'], n_iter=kwargs['TSNE_n_iter'], n_components=2)
        else:
            print("wrong feature reduction method (UMAP, tSNE)")
            assert ValueError
        
        emb_result = emb.fit_transform(motif_df)
        motif_df['emb_1'] = emb_result[:,0]
        motif_df['emb_2'] = emb_result[:,1]
        
        return emb, motif_df[['emb_1', 'emb_2']]


    def _submotif_pdist(self, motif_df, metric):
        motif_dist = pdist(motif_df, metric=metric)
        motif_dist = squareform(motif_dist)
        motif_dist_df = pd.DataFrame(motif_dist, columns=motif_df.index, index=motif_df.index)
        motif_dist_df = motif_dist_df.fillna(0)
        return motif_dist_df


    def _submotif_clustering(self, motif_df, method, kwargs):
        # remove label
        columns = list(motif_df.columns)
        if 'label' in columns:
            columns.remove('label')

        # method
        if method == 'DBSCAN':
            cluster = DBSCAN(eps=kwargs['DBSCAN_eps'], metric=kwargs['DBSCAN_metric'], min_samples=kwargs['DBSCAN_min_samples'])
        elif method == 'HDBSCAN':
            cluster = HDBSCAN(min_cluster_size=kwargs['HDBSCAN_min_cluster_size'], min_samples=kwargs['HDBSCAN_min_samples'])
        elif method == 'Agglomerative':
            cluster = AgglomerativeClustering(affinity=kwargs['Agglomerative_affinity'],
                                              linkage=kwargs['Agglomerative_linkage'],
                                              distance_threshold=kwargs['Agglomerative_distance_threshold'],
                                              n_clusters=kwargs['Agglomerative_n_clusters'])
        else:
            print("wrong clustering method (DBSCAN, HDBSCAN, AgglomerativeClustering)")
            assert ValueError

        # clustering
        arr = motif_df[columns].to_numpy()
        cluster.fit(arr)

        # add label
        label = cluster.labels_
        motif_df['label'] = label
        ##motif_df['label'] = motif_df['label'].apply(lambda x: 'group_%d'%x)

        return cluster, motif_df


    def _mhc_seqlogo(self, alleles, positions, seqlogo_type='probability'):
        seqs = list()
        for allele in alleles:
            seqs.append(''.join(self.mhc_seq_dict[allele][j] for j in positions))
        temp_df = pd.DataFrame(columns=list(self.aa_str))
        seqlogo_df = lm.alignment_to_matrix(sequences=seqs, to_type=seqlogo_type,
                                            characters_to_ignore=".XU", pseudocount=0.01)
        temp_df = pd.concat([temp_df, seqlogo_df], axis=0)
        temp_df = temp_df[list(self.aa_str)]
        temp_df = temp_df.fillna(0.0)
        return temp_df
    
    
    def _group_count(self, df):
        if 'allele_group' not in df.columns:
            df['allele_group'] = [s.split(':')[0] for s in df.index]
        group_df = pd.DataFrame(columns=sorted(list(df['allele_group'].unique())), index=sorted(list(df['label'].unique())))
        for group, sub_df in df.groupby(['label', 'allele_group']):
            group_df.loc[group[0], group[1]] = sub_df.shape[0]
        group_df = group_df.fillna(0.0)
        return group_df


    """"""""""""""""""""""""""""""""""""""
    # Plots
    """"""""""""""""""""""""""""""""""""""
    def _emb_plot(self, df, hue=None, figsize=(4.5, 3.5), figfile=None):
        fig, ax = plt.subplots(1, figsize=figsize, dpi=self.dpi)
        sns.scatterplot(x='emb_1', y='emb_2', hue=hue, data=df, ax=ax, s=10, linewidth=0.3)
        # fontsize
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(self.fontsize)
        # legend
        h, l = ax.get_legend_handles_labels()
        ax.legend(h, l, loc='center left', bbox_to_anchor=(1.1,0,0.5,1), fontsize=self.fontsize,
                  labelspacing=14/len(h), handletextpad=0.1, borderpad=0.1, markerscale=0.8)
        
        fig.tight_layout()
        if figfile:
            fig.savefig(figfile)


    def _main_plot(self, hla, side, df, ylim=4, highlight=False, turn_off_label=False, figfile=None):
        # get group list
        group_list = sorted(list(df['label'].unique()))
        try:
            group_list.remove('group_-1')
        except:
            pass
        group_num = len(group_list)

        # HLA seqlogo
        alleles = list(df.index)
        hla_seqlogo_df = self._mhc_seqlogo(alleles, self.position_dict['selected'])
    
        # figure
        fig, axes = plt.subplots(group_num, 2, figsize=(6, group_num*1.5), dpi=self.dpi,
                                 gridspec_kw={'width_ratios': [1, 3]})
        
        # main
        for i in range(group_num):
            alleles = list(df[df['label']==group_list[i]].index)
            num = len(alleles)
            title = '%s, num=%d'%(group_list[i], num)
            
            # Motif
            self._motif_plot(alleles, side, axes[i][0], ylim=ylim, title=title, turn_off_label=turn_off_label)
            
            # MHC seq
            highlight_pos_dict = dict()
            if highlight:
                ## one HLA gene positions
                highlight_pos_dict['#f2f2f2'] = self.position_dict[hla]
                
                ## two HLA gene positions
                half_intersection_pos = set()
                for j in ['A', 'B', 'C']:
                    if j == hla:
                        continue
                    temp_set = set(self.position_dict[j]) & set(self.position_dict[hla])
                    half_intersection_pos = temp_set | half_intersection_pos
                highlight_pos_dict['#d9d9d9'] = list(half_intersection_pos)
                
                ## three HLA gene positions
                highlight_pos_dict['#bfbfbf'] = self.position_dict['intersection']
            
            self._mhcseq_plot(alleles, self.position_dict['selected'], axes[i][1], highlight_pos_dict=highlight_pos_dict,
                              ylim=1, title=title, diff_df=hla_seqlogo_df, turn_off_label=turn_off_label)
        
        fig.tight_layout()
        if figfile:
            fig.savefig(figfile)
            
    
    def _motif_plot(self, alleles, side, ax, ylim=4, title=None, turn_off_label=False):
        seqlogo_arr = np.zeros((8,20))
        for allele in alleles:
            seqlogo_arr += self.mhc_motif_dict[allele]
        seqlogo_arr /= len(alleles)
        if side == 'N':
            seqlogo_df = pd.DataFrame(seqlogo_arr[:self.submotif_len], columns=list(self.aa_str))
            xticklabels = list(range(1, self.submotif_len+1))
        elif side == 'C':
            seqlogo_df = pd.DataFrame(seqlogo_arr[-self.submotif_len:], columns=list(self.aa_str))
            xticklabels = list(range(-self.submotif_len, 0))
        else: #both
            seqlogo_df = pd.DataFrame(seqlogo_arr, columns=list(self.aa_str))
            xticklabels = list(range(1, self.submotif_len+1)) + list(range(-self.submotif_len, 0))
        logo = lm.Logo(seqlogo_df, color_scheme='skylign_protein', ax=ax)
        _ = ax.set_xticks(list(range(len(xticklabels))))
        _ = ax.set_xticklabels(xticklabels)
        _ = ax.set_ylim(0,ylim)
        _ = ax.set_title(title)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(self.fontsize)
            
        if turn_off_label:
            _ = ax.set_xticks([])
            _ = ax.set_yticks([])
            _ = ax.set_xticklabels([])
            _ = ax.set_yticklabels([])
            _ = ax.set_title(None)
        
    
    def _mhcseq_plot(self, alleles, positions, ax, highlight_pos_dict=dict(),
                     ylim=1, title=None, diff_df=pd.DataFrame(), turn_off_label=False):
        seqlogo_df = self._mhc_seqlogo(alleles, positions)
        if diff_df.shape != (0,0):
            seqlogo_df = seqlogo_df - diff_df

        logo = lm.Logo(seqlogo_df, color_scheme='skylign_protein', ax=ax)
        
        if diff_df.shape != (0,0):
            _ = ax.set_ylim(-ylim, ylim)
        else:
            _ = ax.set_ylim(0, ylim)
        _ = ax.set_xticks(range(len(positions)))
        _ = ax.set_xticklabels([i+1 for i in positions], rotation=90)
        _ = ax.set_title(title)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(self.fontsize)
        for item in ax.get_xticklabels():
            item.set_fontsize(self.fontsize-3)
            
        if turn_off_label:
            _ = ax.set_xticks([])
            _ = ax.set_yticks([])
            _ = ax.set_xticklabels([])
            _ = ax.set_yticklabels([])
            _ = ax.set_title(None)

        if highlight_pos_dict != dict():
            for color, highlight_pos in highlight_pos_dict.items():
                highlight_pos = sorted(set(highlight_pos) & set(positions))
                for pos in highlight_pos:
                    logo.highlight_position(p=positions.index(pos), color=color)

    
    def _group_count_plot(self, df, ax, xticklabel=True):
        sns.heatmap(df, ax=ax, cmap='Blues', linewidths=0.3, cbar=False, annot=True)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(self.fontsize)
        for item in ax.get_yticklabels():
            item.set_rotation(0)
        for item in ax.get_xticklabels():
            item.set_rotation(90)
        if not xticklabel:
            _ = ax.set_xticks([])
            _ = ax.set_xticklabels([])


    def _clustermap_plot(self, df, metric, method, figfile=None):
        labels = df['label']
        colors = ['#ff595e', '#ffca3a', '#98d831', '#1982c4', '#7e5dac', '#a5a3a1', '#ff9a5c', '#b68c68']
        color_dict = dict(zip(labels.unique(), colors))
        label_colors = labels.map(color_dict)

        df = df.drop(columns=['label'])
        g = sns.clustermap(df,
                           metric=metric,
                           method=method,
                           cbar_pos=None,
                           row_colors=label_colors,
                           xticklabels=False,
                           yticklabels=False,
                           dendrogram_ratio=0.1,
                           figsize=(5,5))

        # legend
        for label in sorted(labels.unique()):
            g.ax_row_dendrogram.bar(0, 0, color=color_dict[label], label=label, linewidth=0)
        l = g.ax_row_dendrogram.legend(title='Group', loc='center left', ncol=1, fontsize=self.fontsize,
                                       bbox_to_anchor=(1, 0.5), bbox_transform=plt.gcf().transFigure)

        plt.savefig(figfile, bbox_inches='tight', dpi=self.dpi)
        return g