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
from sklearn.cluster import KMeans
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
        self.group_select_num = 50
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
                   pre_pdist=True, metric='cosine', method='average',
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
        cluster, motif_df = self._submotif_clustering(motif_df, clustering_method, clustering_kwargs)

        # clustermap plot
        g = self._clustermap_plot(motif_df, metric, method, figfile='%s/%s_%ssidePDIST.png'%(self.output_dir, hla, side))
        
        # cluster order
        group_list = list(motif_df.iloc[g.dendrogram_row.reordered_ind]['label'])
        group_list = sorted(set(group_list), key=group_list.index)
        replace_dict = {group_list[i]: i for i in range(len(group_list))}
        replace_dict[-1] = -1
        motif_df = motif_df.replace({'label': replace_dict})
        print("Cluster number:")
        print(motif_df['label'].value_counts())
        
        # add HLA group
        motif_df['group'] = [i.split(':')[0] for i in motif_df.index]
        
        # select alleles
        motif_df['select_label'] = -1
        for label, sub_label_df in motif_df.groupby('label'):
            if sub_label_df.shape[0] < self.group_select_num:
                continue
            temp_alleles = list()
            for group, sub_group_df in sub_label_df.groupby('group'):
                if sub_group_df.shape[0] >= self.group_select_num:
                    temp_alleles += list(sub_group_df.sample(self.group_select_num, replace=False, random_state=0).index)
                else:
                    temp_alleles += list(sub_group_df.index)
            motif_df.loc[temp_alleles, 'select_label'] = label

        # save motif_df
        motif_df.to_csv('%s/%s_%ssideDF.csv'%(self.output_dir, hla, side))
        label_dict =  motif_df['label'].to_dict()
        
        # main plot
        hyper_motif_dict, allele_signature_dict = self._main_plot(hla,
                                                                  side,
                                                                  motif_df,
                                                                  ylim=4,
                                                                  highlight=highlight,
                                                                  turn_off_label=turn_off_label,
                                                                  figfile='%s/%s_%ssideMain.png'%(self.output_dir, hla, side))
        
        # embedding plot
        if reduction_method:
            self._emb_plot(motif_df, hue='label', figfile='%s/%s_%ssideEMB.png'%(self.output_dir, hla, side))
        
        print("%s-side of HLA-%s complete"%(side, hla))

        return label_dict, hyper_motif_dict, allele_signature_dict
    
    
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
                groups = row[row >= self.group_select_num/2].index
                groups = ['%s%s'%(j[0], j[2:]) for j in groups]
                groups_str = ' '.join(groups)
                print(idx, groups_str)
        
        # plot
        fig, ax = plt.subplots(2,1, figsize=(10, 5), dpi=self.dpi)
        self._group_count_plot(Nside_group_df.loc[Nside_group_df.sum(axis=1) > self.group_select_num], ax[0], xticklabel=False)
        self._group_count_plot(Cside_group_df.loc[Cside_group_df.sum(axis=1) > self.group_select_num], ax[1])
        fig.tight_layout()
        fig.savefig('%s/%s_GroupCount.png'%(self.output_dir, hla))
        
        # save
        Nside_group_df.to_csv('%s/%s_NsideGroupCount.csv'%(self.output_dir, hla))
        Cside_group_df.to_csv('%s/%s_CsideGroupCount.csv'%(self.output_dir, hla))
        
        return Nside_group_df, Cside_group_df
        
    
    def Combining(self, hla, interp_dict):
        # build label df
        label_df = pd.DataFrame.from_dict(interp_dict['cluster']['%s_N'%hla], orient='index', columns=['N_terminus'])
        label_df['C_terminus'] = pd.DataFrame.from_dict(interp_dict['cluster']['%s_C'%hla], orient='index', columns=['C_terminus'])['C_terminus']
        label_df['group'] = label_df.index.to_series().apply(lambda x: x.split(':')[0])

        # build empty df
        N_class = sorted(label_df['N_terminus'].unique())
        C_class = sorted(label_df['C_terminus'].unique())

        index_tuple = list()
        for n in N_class:
            for c in C_class:
                index_tuple.append((n, c))

        index = pd.MultiIndex.from_tuples(index_tuple, names=["N_terminus", "C_terminus"])
        column = sorted(label_df['group'].unique())
        comb_df = pd.DataFrame(0, index=index, columns=column)

        # assign value
        for (n,c), row in comb_df.iterrows():
            sub_df = label_df[(label_df['N_terminus']==n) & (label_df['C_terminus']==c)]
            val_df = sub_df['group'].value_counts()
            comb_df.loc[n, c] = val_df
        comb_df.fillna(0, inplace=True)

        # threshold
        ## group threshold: group_num >= 0.5*group_select_num  or group_ratio > group_min_ratio
        group_min_ratio = 0.1
        group_threshold = comb_df.sum(axis=0) * group_min_ratio
        group_threshold[group_threshold > (0.5*self.group_select_num - 1)] = (0.5*self.group_select_num - 1)

        ## combination threshold: comb_num > comb_min and select maximal group
        comb_min = 10
        comb_threshold = comb_df.max(axis=1)
        select_comb_df = comb_df.gt(group_threshold, axis=1) | comb_df.ge(comb_threshold, axis=0)
        comb_min_index = (comb_df.sum(axis=1) > comb_min)

        # assign selected groups to new column 'groups'
        for idx, row in select_comb_df.iterrows():
            if not comb_min_index[idx]:
                select_comb_df.loc[idx, 'groups'] = ""
                continue
            groups = row.index[row].to_list()
            select_comb_df.loc[idx, 'groups'] = " ".join(groups).replace("*", "")
            
        return select_comb_df


    # N_targets = {[cluster]: [color], ...}
    def Demo(self, hla, interp_dict, N_targets, C_targets):
        # selected positions
        positions = self.position_dict['selected']

        # build label df
        label_df = pd.DataFrame.from_dict(interp_dict['cluster']['%s_N'%hla], orient='index', columns=['N_terminus'])
        label_df['C_terminus'] = pd.DataFrame.from_dict(interp_dict['cluster']['%s_C'%hla], orient='index', columns=['C_terminus'])['C_terminus']
        label_df['group'] = label_df.index.to_series().apply(lambda x: x.split(':')[0])

        # demo directory
        if not os.path.isdir('%s/Demo'%self.output_dir):
            os.mkdir('%s/Demo'%self.output_dir)

        # load data and build background mhc seqlogo
        Nside_df = pd.read_csv('%s/%s_NsideDF.csv'%(self.output_dir, hla), index_col=0)
        Cside_df = pd.read_csv('%s/%s_CsideDF.csv'%(self.output_dir, hla), index_col=0)
        hla_seqlogo_df = (self._mhc_seqlogo(Nside_df[Nside_df['select_label']!=-1].index, positions) + \
                          self._mhc_seqlogo(Cside_df[Cside_df['select_label']!=-1].index, positions)) / 2

        # middle position for color boxes
        middle_pos = 76
        n_pos = [i for i in positions if i <= middle_pos]
        c_pos = [i for i in positions if i > middle_pos ]

        # side info
        side_info = {'N': [Nside_df, n_pos, N_targets], 'C': [Cside_df, c_pos, C_targets]}

        # cluster demo
        for side, info in side_info.items():
            side_df, side_pos, side_targets = info
            for cluster, color in side_targets.items():
                fig, ax = plt.subplots(1, 2, figsize=(6, 1.5), dpi=self.dpi, gridspec_kw={'width_ratios': [1, 3]})
                _ = self._motif_plot(side_df[side_df['label']==cluster].index, side, ax[0], turn_off_label=True)
                _ = self._mhcseq_plot(side_df[side_df['select_label']==cluster].index, positions, ax[1],
                                      highlight_pos_dict={color: side_pos}, turn_off_label=True,
                                      diff_df=self._mhc_seqlogo(side_df[side_df['select_label']!=-1].index, positions))
                fig.tight_layout()
                fig.savefig('%s/Demo/ClusterDemo_%s%s%d.png'%(self.output_dir, hla, side, cluster))

        # combination demo
        for n_cluster, n_color in N_targets.items():
            for c_cluster, c_color in C_targets.items():
                highlight_pos_dict = {n_color: n_pos, c_color: c_pos}
                alleles = label_df[(label_df['N_terminus']==n_cluster) & (label_df['C_terminus']==c_cluster)].index
                print('N=%d, C=%d, num=%d'%(n_cluster, c_cluster, len(alleles)))

                # plot
                fig, ax = plt.subplots(1, 2, figsize=(7.5, 1.5), dpi=self.dpi, gridspec_kw={'width_ratios': [2, 3]})
                _ = self._motif_plot(alleles, 'both', ax[0], turn_off_label=True)
                _ = self._mhcseq_plot(alleles, positions, ax[1], diff_df=hla_seqlogo_df,
                                      highlight_pos_dict=highlight_pos_dict, turn_off_label=True)

                fig.tight_layout()
                fig.savefig('%s/Demo/CombineDemo_%sN%dC%d.png'%(self.output_dir, hla, n_cluster, c_cluster))


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
        # remove label column
        cols = list(df.columns)
        new_cols = list()
        for col in cols:
            if ('label' in col) or ('group' in col):
                continue
            new_cols.append(col)

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
        elif method == 'KMeans':
            cluster = KMeans(n_clusters=kwargs['KMeans_n_clusters'], random_state=0)
        else:
            print("wrong clustering method (DBSCAN, HDBSCAN, AgglomerativeClustering)")
            assert ValueError

        # clustering
        arr = motif_df[new_cols].to_numpy()
        cluster.fit(arr)

        # add label
        motif_df['label'] = cluster.labels_

        return cluster, motif_df


    def _mhc_seqlogo(self, alleles, positions, seqlogo_type='counts'):
        seqs = list()
        for allele in alleles:
            seqs.append(''.join(self.mhc_seq_dict[allele][j] for j in positions))
        temp_df = pd.DataFrame(columns=list(self.aa_str))
        seqlogo_df = lm.alignment_to_matrix(sequences=seqs, to_type=seqlogo_type, pseudocount=0) / len(alleles)
        return pd.concat([temp_df, seqlogo_df], axis=0)[temp_df.columns].fillna(0.0)
    
    
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
        cluster_order = sorted(list(df['select_label'].unique()))
        try:
            cluster_order.remove(-1)
        except:
            pass
        group_num = len(cluster_order)

        # HLA seqlogo
        print("background allele number: ", len(list(df[df['select_label']!=-1].index)))
        hla_seqlogo_df = self._mhc_seqlogo(list(df[df['select_label']!=-1].index), self.position_dict['selected'])
    
        # figure
        fig, axes = plt.subplots(group_num, 2, figsize=(6, group_num*1.5), dpi=self.dpi,
                                 gridspec_kw={'width_ratios': [1, 3]})
        
        # main
        hyper_motif_dict = dict()
        allele_signature_dict = dict()

        for i in range(group_num):
            label = cluster_order[i]
            alleles = list(df[df['label']==label].index)
            select_alleles = list(df[df['select_label']==label].index)
            
            if len(select_alleles) == 0:
                continue
            
            # Motif
            hyper_motif = self._motif_plot(alleles,
                                           side,
                                           axes[i][0],
                                           ylim=ylim,
                                           title='group:%s, num=%d'%(label, len(alleles)),
                                           turn_off_label=turn_off_label)
            hyper_motif_dict[label] = hyper_motif.values

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
            
            allele_signature = self._mhcseq_plot(select_alleles,
                                                 self.position_dict['selected'],
                                                 axes[i][1],
                                                 highlight_pos_dict=highlight_pos_dict,
                                                 ylim=1,
                                                 title='group:%s, num=%d'%(label, len(select_alleles)),
                                                 diff_df=hla_seqlogo_df,
                                                 turn_off_label=turn_off_label)
            allele_signature_dict[label] = allele_signature.values
        
        fig.tight_layout()
        if figfile:
            fig.savefig(figfile)

        return hyper_motif_dict, allele_signature_dict
            
    
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

        return seqlogo_df
        
    
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

        return seqlogo_df

    
    def _group_count_plot(self, df, ax, xticklabel=True):
        sns.heatmap(df, ax=ax, cmap='Blues', linewidths=0.3, cbar=False, annot=True, fmt='g')
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
        colors = ['#98d831', '#588b8b', '#ffca3a', '#1982c4', '#7e5dac', '#a5a3a1', '#5fa8d3', '#b68c68']
        color_dict = dict(zip(labels.unique(), colors))
        label_colors = labels.map(color_dict)

        df = df.drop(columns=['label'])
        g = sns.clustermap(df,
                           metric=metric,
                           method=method,
                           cbar_pos=(.3, 0, .4, .02),
                           cbar_kws={'orientation':'horizontal'},
                           row_colors=label_colors.values,
                           xticklabels=False,
                           yticklabels=False,
                           dendrogram_ratio=0.1,
                           figsize=(5,5))
        
        plt.savefig(figfile, bbox_inches='tight', dpi=self.dpi)
        
        return g
